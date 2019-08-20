import os
import math
import torch
from torch.utils.tensorboard import SummaryWriter

from src.asr import ASR
from src.lm import RNNLM
from src.optim import Optimizer
from src.data import load_dataset, load_textset
from src.util import Timer, human_format, cal_er

GRAD_CLIP = 5.0
PROGRESS_STEP = 100      # Std. output refresh freq.
ADDITIONAL_DEV_STEP = 10 # Additional decode steps for objective validation
DEV_N_EXAMPLE = 4        # Number of examples (alignment/text) to show in tensorboard

class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self,config,paras):
        # General Settings
        self.config = config
        self.paras = paras
        self.device = torch.device('cuda') if self.paras.gpu and torch.cuda.is_available() else torch.device('cpu')

        # Name experiment
        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        
        # Filepath setup
        os.makedirs(paras.ckpdir, exist_ok=True)
        self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
        os.makedirs(self.ckpdir, exist_ok=True)

        self.verbose('Loading data... large corpus may took a while.')

    def verbose(self,msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            print('[INFO]',msg)

    def progress(self,msg):
        ''' Verbose function for updating progress on stdout'''
        if self.paras.verbose:
            print('[{}] {}'.format(human_format(self.step),msg).ljust(100),end='\r')
    
    def write_log(self,log_name,log_dict):
        '''Write log to TensorBoard'''
        if type(log_dict) is dict:
            log_dict = {key:val for key, val in log_dict.items() if (val is not None and not math.isnan(val))}

        if log_dict is None:
            pass
        elif len(log_dict)>0:
            if 'align' in log_name or 'spec' in log_name:
                self.log.add_image(log_name,log_dict,self.step)
            elif 'text' in log_name or 'hyp' in log_name:
                self.log.add_text(log_name, log_dict, self.step)
            else:
                self.log.add_scalars(log_name,log_dict,self.step)


class Trainer(Solver):
    ''' Solver for training'''
    def __init__(self,config,paras):
        super(Trainer, self).__init__(config,paras)

        # Logger settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.best_wer = {'att':2.0,'ctc':2.0}

        # Hyperparameters
        self.valid_step = config['hparas']['valid_step']
        self.max_step = config['hparas']['max_step']

        # Init settings
        self.step = 0
        

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt!=0,dim=-1)
        
        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, **self.config['data'])
        for m in msg: self.verbose(m)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        self.asr_model = ASR(self.feat_dim, self.vocab_size, **self.config['model']).to(self.device)
        for m in self.asr_model.create_msg(): self.verbose(m)

        # Losses
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False) # Note: zero_infinity=False is unstable?

        # Optimizer
        self.optimizer = Optimizer(self.asr_model.parameters(),**self.config['hparas'])

        # ToDo: load pre-trained model
        if self.paras.load:
            raise NotImplementedError

        # ToDo: other training methods


    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        timer = Timer()
        ctc_loss = None
        att_loss = None
        timer.set()

        while self.step< self.max_step:
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate = self.optimizer.pre_step(self.step)

                # Fetch data
                feat, feat_len, txt, txt_len = self.fetch_data(data)
                timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                ctc_output, encode_len, att_output, att_align = \
                    self.asr_model( feat, feat_len, max(txt_len),tf_rate=tf_rate,teacher=txt)

                # Compute all objectives
                total_loss = 0
                if ctc_output is not None:
                    if self.paras.ctc_backend =='cudnn':
                        ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), 
                                                 txt.to_sparse().values().to(device=self.device,dtype=torch.int32),
                                                 encode_len.to(device=self.device,dtype=torch.int32),
                                                 txt_len.to(device=self.device,dtype=torch.int32))
                    else:
                        ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), txt, encode_len, txt_len)
                    total_loss += ctc_loss*self.asr_model.ctc_weight
                if att_output is not None:
                    b,t,_ = att_output.shape
                    att_loss = self.seq_loss(att_output.view(b*t,-1),txt.view(-1))
                    # Sum each uttr and devide by length then mean over batch
                    att_loss = torch.mean(torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(txt!=0,dim=-1).float())
                    total_loss += att_loss*(1-self.asr_model.ctc_weight)
                timer.cnt('fw')

                # Backprop
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step '+str(self.step))
                else:
                    self.optimizer.step()
                timer.cnt('bw')

                # Logger
                if self.step%PROGRESS_STEP==0:
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'\
                            .format(total_loss.cpu().item(),grad_norm,timer.show()))
                    self.write_log('loss',{'tr_ctc':ctc_loss})
                    self.write_log('loss',{'tr_att':att_loss})
                    self.write_log('wer',{'tr_att':cal_er(self.tokenizer,att_output,txt),
                                          'tr_ctc':cal_er(self.tokenizer,ctc_output,txt)})
                # Validation
                if self.step%self.valid_step == 0:
                    self.validate()

                # End of step
                self.step+=1
                timer.set()
                if self.step > self.max_step:break

    def validate(self):
        # Eval mode
        self.asr_model.eval()
        dev_wer = {'att':[],'ctc':[]}

        for i,data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len, att_output, att_align = \
                    self.asr_model( feat, feat_len, max(txt_len)+ADDITIONAL_DEV_STEP)

            dev_wer['att'].append(cal_er(self.tokenizer,att_output,txt))
            dev_wer['ctc'].append(cal_er(self.tokenizer,ctc_output,txt))
        
        # Ckpt if performance improves
        for task in ['att','ctc']:
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            if dev_wer[task] < self.best_wer[task]:
                self.best_wer[task] = dev_wer[task]
                self.save_checkpoint('best_{}.pth'.format(task),dev_wer[task])
            self.write_log('wer',{'dv_'+task:dev_wer[task]})

        # Show some example of last batch on tensorboard
        for i in range(min(len(txt),DEV_N_EXAMPLE)):
            if self.step ==0:
                self.write_log('true_text{}'.format(i),self.tokenizer.decode(txt[i].tolist()))
            if att_output is not None:
                self.write_log('att_align{}'.format(i),att_align[i,0,:,:].cpu().unsqueeze(0))
                self.write_log('att_text{}'.format(i),self.tokenizer.decode(att_output[i].argmax(dim=-1).tolist()))
            if ctc_output is not None:
                self.write_log('ctc_text{}'.format(i),self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist()))

        # Resume training
        self.asr_model.train()

    def save_checkpoint(self, f_name, score):
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "asr_model": self.asr_model.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
        }
        torch.save(full_dict, ckpt_path)
        self.verbose("Saved step {} checkpoint (wer = {:.2f}%) and status @ {}".format(human_format(self.step),score*100,ckpt_path))



class LMTrainer(Solver):
    ''' Solver for training language models'''
    def __init__(self,config,paras):
        super(LMTrainer, self).__init__(config,paras)

        # Logger settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.best_loss = 10

        # Hyperparameters
        self.valid_step = config['hparas']['valid_step']
        self.max_step = config['hparas']['max_step']

        # Init settings
        self.step = 0
        
    def fetch_data(self, data):
        ''' Move data to device, insert <sos> and compute text seq. length'''
        txt = torch.cat((torch.zeros((data.shape[0],1),dtype=torch.long),data), dim=1).to(self.device)
        txt_len = torch.sum(data!=0,dim=-1)+1
        return txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.vocab_size, self.tokenizer, msg = \
                         load_textset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, **self.config['data'])
        for m in msg: self.verbose(m)

    def set_model(self):
        ''' Setup ASR model and optimizer '''

        # Model
        self.rnnlm = RNNLM( self.vocab_size, **self.config['model']).to(self.device)
        for m in self.rnnlm.create_msg(): self.verbose(m)
        # Losses
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Optimizer
        self.optimizer = Optimizer(self.rnnlm.parameters(),**self.config['hparas'])
        # ToDo: load pre-trained model
        if self.paras.load:
            raise NotImplementedError

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        timer = Timer()

        while self.step< self.max_step:
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                self.optimizer.pre_step(self.step)

                # Fetch data
                timer.set()
                txt, txt_len = self.fetch_data(data)
                timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                pred, _ = self.rnnlm(txt[:,:-1], txt_len)

                # Compute all objectives

                lm_loss = self.seq_loss(pred.view(-1,self.vocab_size),txt[:,1:].reshape(-1))
                timer.cnt('fw')

                # Backprop
                lm_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.rnnlm.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    self.verbose('Error : grad norm is NaN @ step '+str(self.step))
                else:
                    self.optimizer.step()
                timer.cnt('bw')

                # Logger
                if self.step%PROGRESS_STEP==0:
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'\
                            .format(lm_loss.cpu().item(),grad_norm,timer.show()))
                    self.write_log('entropy',{'tr':lm_loss})
                    self.write_log('perplexity',{'tr':torch.exp(lm_loss).cpu().item()})
                # Validation
                if self.step%self.valid_step == 0:
                    self.validate()

                # End of step
                self.step+=1
                if self.step > self.max_step:break

    
    def validate(self):
        # Eval mode
        self.rnnlm.eval()
        dev_loss = []

        for i,data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(self.dv_set)))
            # Fetch data
            txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                pred, _ = self.rnnlm(txt[:,:-1], txt_len-1)
            lm_loss = self.seq_loss(pred.view(-1,self.vocab_size),txt[:,1:].reshape(-1))
            dev_loss.append(lm_loss)
        
        # Ckpt if performance improves
        dev_loss = sum(dev_loss)/len(dev_loss)
        dev_ppx = torch.exp(dev_loss).cpu().item()
        if dev_loss < self.best_loss :
            self.best_loss = dev_loss
            self.save_checkpoint('best_ppx.pth',dev_ppx)
        self.write_log('perplexity',{'dv':dev_ppx})

        # Show some example of last batch on tensorboard
        for i in range(min(len(txt),DEV_N_EXAMPLE)):
            if self.step ==0:
                self.write_log('true_text{}'.format(i),self.tokenizer.decode(txt[i].tolist()))
            self.write_log('pred_text{}'.format(i),self.tokenizer.decode(pred[i].argmax(dim=-1).tolist()))

        # Resume training
        self.rnnlm.train()

    def save_checkpoint(self, f_name, score):
        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "rnnlm": self.rnnlm.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
        }
        torch.save(full_dict, ckpt_path)
        self.verbose("Saved checkpoint (ppx = {:.2f}) and status @ {}".format(score,ckpt_path))

