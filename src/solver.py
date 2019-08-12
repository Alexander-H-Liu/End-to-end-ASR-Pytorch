import os
import torch
from tensorboardX import SummaryWriter

from src.data import load_dataset
from src.util import human_format

GRAD_CLIP = 5.0

class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self,config,paras):
        # General Settings
        self.config = config
        self.paras = paras
        self.device = torch.device('cuda') if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')

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

    def progress(self,msg,progress_step=100):
        ''' Verbose function for updating progress on stdout'''
        if self.paras.verbose and (self.step%progress_step==0):
            print('[{}] {}'.format(human_format(self.step),msg).ljust(100),end='\r')
    
    def write_log(self,log_name,log_dict):
        '''Write log to TensorBoard'''
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
        self.tf_start = config['hparas']['tf_start']
        self.tf_end = config['hparas']['tf_end']

        # Init settings
        self.step = 0
        

    def fetch_data(self, data):
        # Move to device
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
        pass #ToDo


    def exec(self):
        pass # ToDo
    
    def validate(self):
        pass # ToDo


    def save_checkpoint(self):
        pass # ToDo

