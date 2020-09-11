import copy
import torch
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from src.solver import BaseSolver
from src.asr import ASR
from src.decode import BeamDecoder
from src.ctc import CTCBeamDecoder
from src.data import load_dataset

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)

        # ToDo : support tr/eval on different corpus
        assert self.config['data']['corpus']['name'] == self.src_config['data']['corpus']['name']
        self.config['data']['corpus']['path'] = self.src_config['data']['corpus']['path']
        self.config['data']['corpus']['bucketing'] = False

        # The follow attribute should be identical to training config
        self.config['data']['audio'] = self.src_config['data']['audio']
        self.config['data']['text'] = self.src_config['data']['text']
        self.config['hparas'] = self.src_config['hparas']
        self.config['model'] = self.src_config['model']

        # Output file
        self.output_file = str(self.ckpdir)+'_{}_{}.csv'

        # Override batch size for beam decoding
        self.greedy = self.config['decode']['beam_size'] == 1
        if not self.greedy:
            self.config['data']['corpus']['batch_size'] = 1
        
        self.step = 0
    
    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length,
            For Greedy decoding only ( beam_decode & ctc_beam_decode otherwise)'''
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt!=0,dim=-1)

        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.dv_set, self.tt_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
            load_dataset(self.paras.njobs, self.paras.gpu,
                         self.paras.pin_memory, False, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model '''
        # Model
        init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'
        self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                         self.config['model']).to(self.device)

        # Plug-ins
        if ('emb' in self.config) and (self.config['emb']['enable']) \
                and (self.config['emb']['fuse'] > 0):
            from src.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb'])

        # Load target model in eval mode
        self.load_ckpt()

        self.ctc_only = False
        if self.greedy:
            # Greedy decoding: attention-based if the ASR has a decoder, else use CTC
            self.decoder = copy.deepcopy(self.model).to(self.device)
        else:
            if (not self.model.enable_att) or self.config['decode'].get('ctc_weight', 0.0) == 1.0:
                # Pure CTC Beam Decoder
                assert self.config['decode']['beam_size'] <= self.config['decode']['vocab_candidate']
                self.decoder = CTCBeamDecoder(self.model.to(self.device),
                    [1] + [r for r in range(3, self.vocab_size)],
                    self.config['decode']['beam_size'],
                    self.config['decode']['vocab_candidate'],
                    lm_path=self.config['decode']['lm_path'],
                    lm_config=self.config['decode']['lm_config'],
                    lm_weight=self.config['decode']['lm_weight'],
                    device=self.device)
                self.ctc_only = True
            else:
                # Joint CTC-Attention Beam Decoder
                self.decoder = BeamDecoder(
                    self.model.cpu(), self.emb_decoder, **self.config['decode'])
        
        self.verbose(self.decoder.create_msg())
        del self.model
        del self.emb_decoder

    def greedy_decode(self, dv_set):
        ''' Greedy Decoding '''
        results = []
        for i,data in enumerate(dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)
            # Forward model
            with torch.no_grad():
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.decoder( feat, feat_len, int(float(feat_len.max()) * self.config['decode']['max_len_ratio']), 
                                    emb_decoder=self.emb_decoder)
            for j in range(len(txt)):
                idx = j + self.config['data']['corpus']['batch_size'] * i
                if att_output is not None:
                    hyp_seqs = att_output[j].argmax(dim=-1).tolist()
                else:
                    hyp_seqs = ctc_output[j].argmax(dim=-1).tolist()
                true_txt = txt[j]
                results.append((str(idx), [hyp_seqs], true_txt))
        return results
    
    def exec(self):
        ''' Testing End-to-end ASR system '''
        for s, ds in zip(['dev','test'],[self.dv_set,self.tt_set]):
            # Setup output
            self.cur_output_path = self.output_file.format(s,'output')
            with open(self.cur_output_path,'w',encoding='UTF-8') as f:
                f.write('idx\thyp\ttruth\n')

            if self.greedy:
                # Greedy decode
                self.verbose(
                    'Performing batch-wise greedy decoding on {} set, num of batch = {}.'.format(s, len(ds)))
                results = self.greedy_decode(ds)
                self.verbose('Results will be stored at {}'.format(
                    self.cur_output_path))
                self.write_hyp(results, self.cur_output_path, '-')
            elif self.ctc_only:
                # CTC beam decode
                # Additional output to store all beams
                self.cur_beam_path = self.output_file.format(s,
                    'beam-{}-{}'.format(self.config['decode']['beam_size'], self.config['decode']['lm_weight']))
                with open(self.cur_beam_path,'w') as f:
                    f.write('idx\tbeam\thyp\ttruth\n')
                self.verbose('Performing instance-wise CTC beam decoding on {} set, num of batch = {}.'.format(s,len(ds)))
                # Minimal function to pickle
                ctc_beam_decode_func = partial(ctc_beam_decode, model=copy.deepcopy(self.decoder), device=self.device)
                # Parallel beam decode
                results = Parallel(n_jobs=self.paras.njobs)(delayed(ctc_beam_decode_func)(data) for data in tqdm(ds))
                self.verbose('Results/Beams will be stored at {} / {}'.format(self.cur_output_path,self.cur_beam_path))
                self.write_hyp(results, self.cur_output_path, self.cur_beam_path)
            else:
                # Joint CTC-Attention beam decode
                # Additional output to store all beams
                self.cur_beam_path = self.output_file.format(s,
                    'beam-{}-{}'.format(self.config['decode']['beam_size'], self.config['decode']['lm_weight']))
                with open(self.cur_beam_path, 'w') as f:
                    f.write('idx\tbeam\thyp\ttruth\n')
                self.verbose(
                    'Performing instance-wise beam decoding on {} set. (NOTE: use --njobs to speedup)'.format(s))
                # Minimal function to pickle
                beam_decode_func = partial(beam_decode, model=copy.deepcopy(
                    self.decoder), device=self.device)
                # Parallel beam decode
                results = Parallel(n_jobs=self.paras.njobs)(
                    delayed(beam_decode_func)(data) for data in tqdm(ds))
                self.verbose(
                    'Results/Beams will be stored at {} / {}.'.format(self.cur_output_path, self.cur_beam_path))
                self.write_hyp(results, self.cur_output_path,
                               self.cur_beam_path)
        self.verbose('All done !')

    def write_hyp(self, results, best_path, beam_path):
        '''Record decoding results'''
        if self.greedy:
            # Ignores repeated symbols if is decoded with CTC
            ignore_repeat = not self.decoder.enable_att
        else:
            ignore_repeat = False
        for name, hyp_seqs, truth in tqdm(results):
            if self.ctc_only and not self.greedy:
                new_hyp_seqs = [self.tokenizer.decode(hyp, ignore_repeat=False) for hyp in hyp_seqs[:-1]]
                hyp_seqs = new_hyp_seqs + [self.tokenizer.decode(hyp_seqs[-1], ignore_repeat=True)]
            else:
                hyp_seqs = [self.tokenizer.decode(hyp) for hyp in hyp_seqs]
            
            truth = self.tokenizer.decode(truth)
            with open(best_path, 'a') as f:
                if len(hyp_seqs[0]) == 0:
                    # Set the sequence to a whitespace if it was empty
                    hyp_seqs[0] = ' '
                f.write('\t'.join([name, hyp_seqs[0], truth])+'\n')
            if not self.greedy:
                with open(beam_path, 'a', encoding='UTF-8') as f:
                    for b, hyp in enumerate(hyp_seqs):
                        f.write('\t'.join([name, str(b), hyp, truth])+'\n')

def beam_decode(data, model, device):
    # Fetch data : move data/model to device
    name, feat, feat_len, txt = data
    feat = feat.to(device)
    feat_len = feat_len.to(device)
    txt = txt.to(device)
    txt_len = torch.sum(txt != 0, dim=-1)
    model = model.to(device)
    # Decode
    with torch.no_grad():
        hyps = model(feat, feat_len)

    hyp_seqs = [hyp.outIndex for hyp in hyps]
    del hyps
    return (name[0], hyp_seqs, txt[0].cpu().tolist())  # Note: bs == 1

def ctc_beam_decode(data, model, device):
    # Fetch data : move data/model to device
    name, feat, feat_len, txt = data
    feat = feat.to(device)
    feat_len = feat_len.to(device)
    # Decode
    with torch.no_grad():
        hyp = model(feat, feat_len)

    return (name[0], hyp, txt[0]) # Note: bs == 1
