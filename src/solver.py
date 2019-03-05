import os
import torch
import copy
import math
import numpy as np
import itertools
from tensorboardX import SummaryWriter
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn.functional as F
from src.asr import Seq2Seq
from src.rnnlm import RNN_LM
from src.clm import CLM_wrapper
from src.dataset import LoadDataset
from src.postprocess import Mapper,cal_acc,cal_cer,draw_att,human_format

# TODO : Move these to config
VAL_STEP = 30        # Additional Inference Timesteps to run during validation (to calculate CER)
TRAIN_WER_STEP = 500 # steps for debugging info.
GRAD_CLIP = 5
CLM_MIN_SEQ_LEN = 5

class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self,config,paras):
        # General Settings
        self.config = config
        self.paras = paras
        self.device = torch.device('cuda') if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')

        # Name experiments (if none, automatically named after config file)
        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        
        # Setup directories 
        if not os.path.exists(paras.ckpdir):
            os.makedirs(paras.ckpdir)
        self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
        if not os.path.exists(self.ckpdir):
            os.makedirs(self.ckpdir)
        
        # Load Mapper for idx2token
        self.mapper = Mapper(config['solver']['data_path'])

    def verbose(self,msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            print('[INFO] {}'.format(msg).ljust(100))
   
    def progress(self,msg):
        ''' Verbose function for updating progress on stdout'''
        if self.paras.verbose and self.step%TRAIN_WER_STEP==0:
            print('[{}] {}'.format(human_format(self.step),msg).ljust(100),end='\r')

    def fetch_data(self,x,y):
        '''Unbucket batch and compute input length & label length'''
        if len(x.shape)==4: x = x.squeeze(0)
        if len(y.shape)==3: y = y.squeeze(0)
        x = x.to(device = self.device,dtype=torch.float32)
        y = y.to(device = self.device,dtype=torch.long)
        state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
        state_len = [int(sl) for sl in state_len]
        ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))
        label = y[:,1:ans_len+1].contiguous()
        
        return x,y,state_len,ans_len,label


class Trainer(Solver):
    ''' Handler for complete training progress'''
    def __init__(self,config,paras):
        super(Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['dev_step']
        self.best_val_ed = 2.0

        # Training details
        self.step = 0
        self.max_step = config['solver']['total_steps']
        self.tf_start = config['solver']['tf_start']
        self.tf_end = config['solver']['tf_end']
        self.apex = config['solver']['apex']

        # CLM option
        self.apply_clm = config['clm']['enable']

    def load_data(self):
        ''' Load date for training/validation'''
        self.verbose('Loading data from '+self.config['solver']['data_path'])
        setattr(self,'train_set',LoadDataset('train',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        
        # Get 1 example for auto constructing model
        for self.sample_x,_ in getattr(self,'train_set'):break
        if len(self.sample_x.shape)==4: self.sample_x=self.sample_x[0]

    def set_model(self):
        ''' Setup ASR (and CLM if enabled)'''
        self.verbose('Init ASR model. Note: validation is done through greedy decoding w/ attention decoder.')
        
        # Build attention end-to-end ASR
        self.asr_model = Seq2Seq(self.sample_x,self.mapper.get_dim(),self.config['asr_model']).to(self.device)
        if 'VGG' in self.config['asr_model']['encoder']['enc_type']:
            self.verbose('VCC Extractor in Encoder is enabled, time subsample rate = 4.')
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')
        
        # Involve CTC
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        self.ctc_weight = self.config['asr_model']['optimizer']['joint_ctc']
        self.enable_ctc = self.ctc_weight > 0
        self.enable_att = self.ctc_weight < 1
        
        # TODO: load pre-trained model
        if self.paras.load:
            raise NotImplementedError
            
        # Setup optimizer
        if self.apex and self.config['asr_model']['optimizer']['type']=='Adam':
            import apex
            self.asr_opt = apex.optimizers.FusedAdam(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'])
        else:
            self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)

        # Apply CLM
        if self.apply_clm:
            self.clm = CLM_wrapper(self.mapper.get_dim(), self.config['clm']).to(self.device)
            clm_data_config = self.config['solver']
            clm_data_config['train_set'] = self.config['clm']['source']
            clm_data_config['use_gpu'] = self.paras.gpu
            self.clm.load_text(clm_data_config)
            self.verbose('CLM is enabled with text-only source: '+str(clm_data_config['train_set']))
            self.verbose('Extra text set total '+str(len(self.clm.train_set))+' batches.')

    def exec(self):
        ''' Training End-to-end ASR system'''
        self.verbose('Training set total '+str(len(self.train_set))+' batches.')
        tr_ter = 1.0

        while self.step< self.max_step:
            for x,y in self.train_set:

                # Init
                loss_log = {}
                ctc_loss,att_loss = 0,0
                tf_rate = self.tf_start - self.step*(self.tf_start-self.tf_end)/self.max_step

                # Fetch data
                x,y,state_len,ans_len,label = self.fetch_data(x,y)

                # ASR forwarding
                self.asr_opt.zero_grad()
                ctc_pred, state_len, att_pred, _ =  self.asr_model(x, ans_len,tf_rate=tf_rate,teacher=y,state_len=state_len)                
                
                # CE loss on attention decoder
                if self.enable_att:
                    b,t,c = att_pred.shape
                    att_loss = self.seq_loss(att_pred.view(b*t,c),label.view(-1))
                    att_loss = torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                               .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                    att_loss = torch.mean(att_loss) # Mean by batch
                    loss_log['train_att'] = att_loss

                # CTC loss on CTC decoder
                if self.enable_ctc:
                    target_len = torch.sum(y!=0,dim=-1)
                    ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, torch.LongTensor(state_len), target_len)
                    loss_log['train_ctc'] = ctc_loss
                
                # Combine CTC/attention loss
                asr_loss = (1-self.ctc_weight)*att_loss+self.ctc_weight*ctc_loss
                loss_log['train_full'] = asr_loss
                
                # Adversarial loss from CLM
                if self.apply_clm and att_pred.shape[1]>=CLM_MIN_SEQ_LEN:
                    if (self.step%self.clm.update_freq)==0:
                        # update CLM every N steps
                        clm_log,gp = self.clm.train(att_pred.detach(),CLM_MIN_SEQ_LEN)
                        self.write_log('clm_score',clm_log)
                        self.write_log('clm_gp',gp)
                    adv_feedback = self.clm.compute_loss(F.softmax(att_pred))
                    asr_loss -= adv_feedback

                # Check grad. norm and backprop
                asr_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step '+str(self.step))
                else:
                    self.asr_opt.step()
                
                # Logger
                self.write_log('loss',loss_log)
                if self.enable_att:
                    # Frame-wise accuraccy (for reference only) 
                    self.write_log('acc',{'train':cal_acc(att_pred,label)})
                if self.step % TRAIN_WER_STEP ==0:
                    # token error rate is calculated during training (for reference only)  
                    tr_ter = cal_cer(att_pred,label,mapper=self.mapper)
                    self.write_log('error rate',
                                   {'train':tr_ter})
                self.progress('Training status | Loss - {:.4f} | Grad. Norm - {:.4f} | Token Error Rate - {:.4f}'\
                    .format(loss_log['train_full'].item(),grad_norm,tr_ter))

                # Validation
                if self.step%self.valid_step == 0:
                    self.asr_opt.zero_grad()
                    self.valid()

                # End of step
                self.step+=1
                if self.step > self.max_step:break
    

    def write_log(self,val_name,val_dict):
        '''Write log to TensorBoard'''
        if 'att' in val_name:
            self.log.add_image(val_name,val_dict,self.step)
        elif 'txt' in val_name or 'hyp' in val_name:
            self.log.add_text(val_name, val_dict, self.step)
        else:
            self.log.add_scalars(val_name,val_dict,self.step)


    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding with Attention decoder only)'''
        self.asr_model.eval()
        
        # Init stats
        val_loss, val_ctc_loss, val_att_loss, val_acc, val_att_er,val_ctc_er = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_cnt = 0
        full_att_pred,full_ctc_pred,full_truth = [],[],[]
        
        # Perform validation
        for cur_b,(x,y) in enumerate(self.dev_set):
            self.progress(' '.join(['Valid step -',str(cur_b),'/',str(len(self.dev_set))]))

            # Fetch data
            x,y,state_len,ans_len,label = self.fetch_data(x,y)
            batch_size = int(x.shape[0])
            
            # Forward
            ctc_pred, state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)

            # Compute attention loss & get decoding results
            if self.enable_att:
                seq_loss = self.seq_loss(att_pred[:,:ans_len,:].contiguous().view(-1,att_pred.shape[-1]),label.view(-1))
                seq_loss = torch.sum(seq_loss.view(x.shape[0],-1),dim=-1)/torch.sum(y!=0,dim=-1)\
                           .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                seq_loss = torch.mean(seq_loss) # Mean by batch
                val_att_loss += seq_loss.detach().cpu()*batch_size
                pred,truth = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                full_att_pred += pred
                full_truth += truth
                val_acc += cal_acc(att_pred,label)*batch_size
                val_att_er += cal_cer(att_pred,label,mapper=self.mapper)*batch_size
            
            # Compute CTC loss & get decoding results
            if self.enable_ctc:
                target_len = torch.sum(y!=0,dim=-1)
                ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, 
                                         torch.LongTensor(state_len), target_len)
                val_ctc_loss += ctc_loss.detach().cpu()*batch_size
                pred,truth = cal_cer(ctc_pred,label,mapper=self.mapper,get_sentence=True)
                full_ctc_pred += pred
                if not self.enable_att:
                    full_truth += truth
                val_ctc_er += cal_cer(ctc_pred,label,mapper=self.mapper)*batch_size

            val_cnt += batch_size
        
        # Logger
        val_loss = (1-self.ctc_weight)*val_att_loss + self.ctc_weight*val_ctc_loss
        loss_log = {}
        for k,v in zip(['dev_full','dev_ctc','dev_att'],[val_loss, val_ctc_loss, val_att_loss]):
            if v > 0.0:
                loss_log[k] = v/val_cnt
        self.write_log('loss',loss_log)
 
        if self.enable_att:
            # Plot attention map to log
            val_hyp,val_txt = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
            val_attmap = draw_att(att_maps,att_pred)

            # Record loss
            self.write_log('error rate',{'dev_att':val_att_er/val_cnt})
            self.write_log('acc',{'dev':val_acc/val_cnt})
            for idx,attmap in enumerate(val_attmap):
                self.write_log('att_'+str(idx),attmap)
                self.write_log('hyp_'+str(idx),val_hyp[idx])
                self.write_log('txt_'+str(idx),val_txt[idx])

            # Save model by val er.
            if val_att_er/val_cnt  < self.best_val_ed:
                self.best_val_ed = val_att_er/val_cnt
                self.verbose('Best val error rate  : {:.4f}       @ step {} {}'.format(self.best_val_ed,self.step,' '*20))
                torch.save(self.asr_model, os.path.join(self.ckpdir,'asr'))
                if self.apply_clm:
                    torch.save(self.clm.clm,  os.path.join(self.ckpdir,'clm'))
                # Save hyps.
                with open(os.path.join(self.ckpdir,'best_hyp.txt'),'w') as f:
                    for pred,truth in zip(full_att_pred,full_truth):
                        f.write(pred+','+truth+'\n')
        
        if self.enable_ctc:
            self.write_log('error rate',{'dev_ctc':val_ctc_er/val_cnt})
            if not self.enable_att and (val_ctc_er/val_cnt  < self.best_val_ed):
                self.best_val_ed = val_ctc_er/val_cnt
                self.verbose('Best val error rate  : {:.4f}       @ step {} {}'.format(self.best_val_ed,self.step,' '*20))
                torch.save(self.asr_model, os.path.join(self.ckpdir,'asr'))
                # Save hyps.
                with open(os.path.join(self.ckpdir,'best_hyp.txt'),'w') as f:
                    for pred,truth in zip(full_ctc_pred,full_truth):
                        f.write(pred+','+truth+'\n')

        self.asr_model.train()


class Tester(Solver):
    ''' Handler for complete inference progress'''
    def __init__(self,config,paras):
        super(Tester, self).__init__(config,paras)
        self.verbose('During beam decoding, batch size is set to 1, please speed up with --njobs.')
        self.njobs = self.paras.njobs
        self.decode_step_ratio = config['solver']['max_decode_step_ratio']
        
        self.decode_file = "_".join(['decode','beam',str(self.config['solver']['decode_beam_size']),
                                     'len',str(self.config['solver']['max_decode_step_ratio'])])

    def load_data(self):
        self.verbose('Loading testing data '+str(self.config['solver']['test_set'])\
                     +' from '+self.config['solver']['data_path'])
        setattr(self,'test_set',LoadDataset('test',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,use_gpu=self.paras.gpu,**self.config['solver']))

    def set_model(self):
        ''' Load saved ASR'''
        self.verbose('Load ASR model from '+os.path.join(self.ckpdir))
        self.asr_model = torch.load(os.path.join(self.ckpdir,'asr'))

        # Decide decoding mode
        if not self.asr_model.joint_att or (self.config['solver']['decode_ctc_weight']==1):
            assert self.asr_model.joint_ctc, "The ASR was not trained with CTC"
            self.decode_mode = 'ctc'
            self.decode_file = "_".join(['decode','ctc'])
        else:
            self.decode_mode = 'attention'

        # Enable joint CTC decoding if needed
        if self.decode_mode == 'attention':
            self.asr_model.clear_att()
            if self.config['solver']['decode_ctc_weight'] >0:
                assert self.asr_model.joint_ctc, "The ASR was not trained with CTC"
                self.decode_mode = 'hybrid'
                self.verbose('Joint CTC decoding is enabled with weight = '+str(self.config['solver']['decode_ctc_weight']))
                self.decode_file += '_ctc{:}'.format(self.config['solver']['decode_ctc_weight'])
                self.asr_model.ctc_weight = self.config['solver']['decode_ctc_weight']
                
        # Enable joint RNNLM decoding
        self.decode_lm = (self.config['solver']['decode_lm_weight']>0) and (self.decode_mode != 'ctc')
        setattr(self.asr_model,'decode_lm_weight',self.config['solver']['decode_lm_weight'])
        if self.decode_lm:
            assert os.path.exists(self.config['solver']['decode_lm_path']), 'Please specify RNNLM path.'
            self.asr_model.load_lm(**self.config['solver'])
            self.verbose('Joint RNNLM decoding is enabled with weight = '+str(self.config['solver']['decode_lm_weight']))
            self.verbose('Loading RNNLM from '+self.config['solver']['decode_lm_path'])
            self.decode_file += '_lm{:}'.format(self.config['solver']['decode_lm_weight'])
        

        # Check models dev performance before inference
        self.asr_model.eval()
        self.asr_model = self.asr_model.to(self.device)
        self.verbose('Checking models performance on dev set '+str(self.config['solver']['dev_set'])+'...')
        self.valid()
        self.asr_model = self.asr_model.to('cpu') # move origin model to cpu, clone it to GPU for each thread

    def exec(self):
        '''Perform inference step with beam search decoding.'''
        test_cer = 0.0
        self.decode_beam_size = self.config['solver']['decode_beam_size']
        self.verbose('Start decoding with beam search, beam size = '+str(self.config['solver']['decode_beam_size']))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'.format(len(self.test_set),self.njobs))
        ## self.test_set = [(x,y) for (x,y) in self.test_set][::10]
        
        if self.decode_mode == 'ctc':
            # Greedy decode for CTC system
            for x,y in tqdm(self.test_set):
                self.ctc_decode(x,y)
            self.verbose('Decode done, results at {}.'.format(str(os.path.join(self.ckpdir,self.decode_file+'.txt'))))
        else:
            # Beam decode for attention/hybrid systems
            _ = Parallel(n_jobs=self.njobs)(delayed(self.beam_decode)(x[0],y[0].tolist()[0]) for x,y in tqdm(self.test_set))
            self.verbose('Decode done, best results at {}.'.format(str(os.path.join(self.ckpdir,self.decode_file+'.txt'))))
            self.verbose('Top {} results at {}.'.format(self.config['solver']['decode_beam_size'],
                                                    str(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'))))
        
    def write_hyp(self,hyps,y):
        '''Record decoding results'''

        if self.decode_mode == 'ctc':
            # CTC decode performed batch-wised
            with open(os.path.join(self.ckpdir,self.decode_file+'.txt'),'a') as f:
                for gt,pd in zip(y,hyps):
                    gt = self.mapper.translate(gt,return_string=True)
                    pd = self.mapper.translate(pd,return_string=True)
                    f.write(gt+'\t'+pd+'\n')
        else:
            gt = self.mapper.translate(y,return_string=True)
            # Best
            with open(os.path.join(self.ckpdir,self.decode_file+'.txt'),'a') as f:
                best_hyp = self.mapper.translate(hyps[0].outIndex,return_string=True)
                f.write(gt+'\t'+best_hyp+'\n')
            # N best
            with open(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'),'a') as f:
                for hyp in hyps:
                    best_hyp = self.mapper.translate(hyp.outIndex,return_string=True)
                    f.write(gt+'\t'+best_hyp+'\n')
        
    def ctc_decode(self,x,y):
        '''Perform batch-wise CTC decoding'''
        # Fetch data
        x,y,state_len,ans_len,label = self.fetch_data(x,y)

        # Forward
        with torch.no_grad():
            ctc_pred, _, _, _ = self.asr_model(x, ans_len,state_len=state_len)

        self.write_hyp(ctc_pred,y)

    def beam_decode(self,x,y):
        '''Perform beam decoding with end-to-end ASR'''
        # Prepare data
        x = x.to(device = self.device,dtype=torch.float32)
        state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
        state_len = [int(sl) for sl in state_len]

        # Forward
        with torch.no_grad():
            max_decode_step =  int(np.ceil(state_len[0]*self.decode_step_ratio))
            model = copy.deepcopy(self.asr_model).to(self.device)
            hyps = model.beam_decode(x, max_decode_step, state_len, self.decode_beam_size)
        del model
        
        self.write_hyp(hyps,y)
        del hyps
        
        return 1

    
    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding on Attention decoder only)'''
        val_att_er,val_ctc_er = 0.0,0.0
        val_cnt = 0    
        full_att_pred,full_ctc_pred,full_truth = [],[],[]
        ctc_results = []
        with torch.no_grad():
            for cur_b,(x,y) in enumerate(self.dev_set):
                self.progress(' '.join(['Valid step - (',str(cur_b),'/',str(len(self.dev_set)),')']))

                # Fetch data
                x,y,state_len,ans_len,label = self.fetch_data(x,y)

                # Forward
                ctc_pred, state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)
                
                # Attention result
                if self.decode_mode in ['attention','hybrid']:
                    pred,truth = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                    full_att_pred += pred
                    full_truth += truth
                    val_att_er += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])

                # CTC result
                if self.decode_mode in ['ctc','hybrid']:
                    pred,truth = cal_cer(ctc_pred,label,mapper=self.mapper,get_sentence=True)
                    full_ctc_pred += pred
                    if self.decode_mode == 'ctc':
                        full_truth += truth
                    val_ctc_er += cal_cer(ctc_pred,label,mapper=self.mapper)*int(x.shape[0])
                                
                val_cnt += int(x.shape[0])
        
        
        # Dump att model score to ensure model is corrected
        if self.decode_mode in ['attention','hybrid']:
            self.verbose('Validation Error Rate of attention decoder : {:.4f}      '.format(val_att_er/val_cnt)) 
            self.verbose('See {} for results.'.format(os.path.join(self.ckpdir,'dev_att_decode.txt'))) 
            with open(os.path.join(self.ckpdir,'dev_att_decode.txt'),'w') as f:
                for hyp,gt in zip(full_att_pred,full_truth):
                    f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')
        
        # Dump ctc model score to ensure model is corrected
        if self.decode_mode in ['ctc','hybrid']:
            self.verbose('Validation Error Rate of CTC decoder : {:.4f}      '.format(val_ctc_er/val_cnt))
            self.verbose('See {} for results.'.format(os.path.join(self.ckpdir,'dev_ctc_decode.txt'))) 
            with open(os.path.join(self.ckpdir,'dev_ctc_decode.txt'),'w') as f:
                for hyp,gt in zip(ctc_text,full_truth):
                    f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')


class RNNLM_Trainer(Solver):
    ''' Trainer for RNN-LM only'''
    def __init__(self, config, paras):
        super(RNNLM_Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['dev_step']
        self.best_dev_ppx = 1000

        # training details
        self.step = 0
        self.max_step = config['solver']['total_steps']
        self.apex = config['solver']['apex']

    def load_data(self):
        ''' Load training / dev set'''
        self.verbose('Loading text data from '+self.config['solver']['data_path'])
        setattr(self,'train_set',LoadDataset('train',text_only=True,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=True,use_gpu=self.paras.gpu,**self.config['solver']))

    def set_model(self):
        ''' Setup RNNLM'''
        self.verbose('Init RNNLM model.')
        self.rnnlm = RNN_LM(out_dim=self.mapper.get_dim(),**self.config['rnn_lm']['model_para'])
        self.rnnlm = self.rnnlm.to(self.device)

        if self.paras.load:
            raise NotImplementedError

        # optimizer
        if self.apex and self.config['rnn_lm']['optimizer']['type']=='Adam':
            import apex
            self.rnnlm_opt = apex.optimizers.FusedAdam(self.rnnlm.parameters(), lr=self.config['rnn_lm']['optimizer']['learning_rate'])
        else:
            self.rnnlm_opt = getattr(torch.optim,self.config['rnn_lm']['optimizer']['type'])
            self.rnnlm_opt = self.rnnlm_opt(self.rnnlm.parameters(), lr=self.config['rnn_lm']['optimizer']['learning_rate'],eps=1e-8)

    def exec(self):
        ''' Training RNN-LM'''
        self.verbose('RNN-LM Training set total '+str(len(self.train_set))+' batches.')

        while self.step < self.max_step:
            for y in self.train_set:
                self.progress('Training step - '+str(self.step))
                # Load data
                if len(y.shape)==3: y = y.squeeze(0)
                y = y.to(device = self.device,dtype=torch.long)
                ans_len = torch.sum(y!=0,dim=-1)

                self.rnnlm_opt.zero_grad()
                _, prob = self.rnnlm(y[:,:-1],ans_len)
                loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
                loss.backward()
                self.rnnlm_opt.step()

                # logger
                ppx = torch.exp(loss.cpu()).item()
                self.log.add_scalars('perplexity',{'train':ppx},self.step)

                # Next step
                self.step += 1
                if self.step % self.valid_step ==0:
                    self.valid()
                if self.step > self.max_step:
                    break

    def valid(self):
        self.rnnlm.eval()

        print_loss = 0.0
        dev_size = 0 

        for cur_b,y in enumerate(self.dev_set):
            self.progress(' '.join(['Valid step -',str(cur_b),'/',str(len(self.dev_set))]))
            if len(y.shape)==3: y = y.squeeze(0)
            y = y.to(device = self.device,dtype=torch.long)
            ans_len = torch.sum(y!=0,dim=-1)
            _, prob = self.rnnlm(y[:,:-1],ans_len)
            loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
            print_loss += loss.clone().detach() * y.shape[0]
            dev_size += y.shape[0]

        print_loss /= dev_size
        dev_ppx = torch.exp(print_loss).cpu().item()
        self.log.add_scalars('perplexity',{'dev':dev_ppx},self.step)
        
        # Store model with the best perplexity
        if dev_ppx < self.best_dev_ppx:
            self.best_dev_ppx  = dev_ppx
            self.verbose('Best val ppx      : {:.4f}       @ step {}'.format(self.best_dev_ppx,self.step))
            torch.save(self.rnnlm,os.path.join(self.ckpdir,'rnnlm'))

        self.rnnlm.train()

class RNNLM_Tester(Solver):
    ''' Tester for RNN-LM only'''
    def __init__(self, config, paras):
        super(RNNLM_Tester, self).__init__(config,paras)

    def load_data(self):
        ''' Load testing set'''
        self.verbose('Loading text data from '+self.config['solver']['data_path'])
        setattr(self,'test_set',LoadDataset('test',text_only=True,use_gpu=self.paras.gpu,**self.config['solver']))

    def set_model(self):
        ''' Load saved LM '''
        self.verbose('Load LM model from ' + os.path.join(self.ckpdir))
        self.rnnlm = torch.load(os.path.join(self.ckpdir, 'rnnlm'))
        self.rnnlm = self.rnnlm.to(self.device)
        self.rnnlm.eval()

    def exec(self):
        self.rnnlm.eval()

        print_loss = 0.0
        dev_size = 0 

        with torch.no_grad():
            for cur_b,y in enumerate(self.test_set):
                self.progress(' '.join(['Testing step -', '(',str(cur_b),'/',str(len(self.test_set)),')']))
                if len(y.shape)==3: y = y.squeeze(0)
                y = y.to(device = self.device,dtype=torch.long)
                ans_len = torch.sum(y!=0,dim=-1)
                _, prob = self.rnnlm(y[:,:-1],ans_len)
                loss = F.cross_entropy(prob.view(-1,prob.shape[-1]), y[:,1:].contiguous().view(-1), ignore_index=0)
                print_loss += loss.clone().detach() * y.shape[0]
                dev_size += y.shape[0]

        print_loss /= dev_size
        dev_ppx = torch.exp(print_loss).cpu().item()
        self.verbose('perplexity: {}'.format(dev_ppx))
