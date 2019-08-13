import os
import math
import torch
from torch.utils.tensorboard import SummaryWriter

from src.asr import ASR
from src.optim import Optimizer
from src.data import load_dataset
from src.util import Timer, human_format, cal_er

GRAD_CLIP = 5.0
PROGRESS_STEP = 100

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
        for k,v in log_dict.items(): 
            if math.isnan(v) or v is None:
                del log_dict[k]
        if len(log_dict)>0:
            if 'align' in log_name or 'spec' in log_name:
                self.log.add_image(log_name,log_dict,self.step)
            elif 'txt' in log_name or 'hyp' in log_name:
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
        self.best_er = 2.0

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
        txt_len = torch.sum(utt_txt!=0,dim=-1)
        
        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = load_dataset(self.config['data'])
        for m in msg: self.verbose(m)


    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        self.asr_model = ASR(self.feat_dim, self.vocab_size, **self.config['model']).to(self.device)
        for m in self.asr_model.create_msg(): self.verbose(m)

        # Losses
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)
        self.ctc_loss = torch.nn.CTCLoss(blank=0)

        # Optimizer
        self.optimizer = Optimizer(self.asr_model.parameters(),**self.config['hparas'])

        # ToDo: load pre-trained model
        if self.paras.load:
            raise NotImplementedError

        # ToDo: other training methods


    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {} ({} steps per epoch).'.format(human_format(self.max_step),len(self.tr_set)))
        timer = Timer()
        ctc_loss = None
        att_loss = None

        while self.step< self.max_step:
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate = self.optimizer.pre_step(self.step)

                # Fetch data
                timer.set()
                feat, feat_len, txt, txt_len = self.fetch_data(data)
                timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                ctc_output, encode_len, att_output, att_salign = \
                    self.asr_model( feat, feat_len, max(txt_len),tf_rate=0.0,teacher=txt)

                # Compute all objectives
                total_loss = 0
                if ctc_output is not None:
                    ctc_loss = self.ctc_loss(ctc_output, txt, encode_len, txt_len)
                    total_loss += ctc_loss*self.asr_model.ctc_weight
                if att_output is not None:
                    b,t = att_loss.shape
                    att_loss = self.seq_loss(att_output.view(b*t,-1),txt.view(-1))
                    # Sum each uttr and devide by length then mean over batch
                    att_loss = torch.mean(torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(txt!=0,dim=-1))
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
                    self.write_log('ctc_loss',{'tr':ctc_loss.cpu().item()})
                    self.write_log('att_loss',{'tr':att_loss.cpu().item()})
                    self.write_log('wer',{'tr_att':cal_er(self.tokenizer,att_output,txt),
                                          'tr_ctc':cal_er(self.tokenizer,ctc_output,txt)})
                # Validation
                if self.step%self.valid_step == 0:
                    self.validate()

                # End of step
                self.step+=1
                if self.step > self.max_step:break

    
    def validate(self):
        self.asr_model.eval()
        # ToDo
        self.asr_model.train()

    def save_checkpoint(self):
        pass # ToDo

