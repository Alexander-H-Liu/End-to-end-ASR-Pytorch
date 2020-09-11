import yaml
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F

from src.lm import RNNLM

LOG_ZERO = -10000000.0  # Log-zero for CTC

class CTCPrefixScore():
    ''' 
    CTC Prefix score calculator
    An implementation of Algo. 2 in https://www.merl.com/publications/docs/TR2017-190.pdf (Watanabe et. al.)
    Reference (official implementation): https://github.com/espnet/espnet/tree/master/espnet/nets
    '''

    def __init__(self, x):
        self.logzero = -100000000.0
        self.blank = 0
        self.eos = 1
        self.x = x.cpu().numpy()[0]
        self.odim = x.shape[-1]
        self.input_length = len(self.x)

    def init_state(self):
        # 0 = non-blank, 1 = blank
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)

        # Accumalate blank at each step
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i-1, 1] + self.x[i, self.blank]
        return r

    def full_compute(self, g, r_prev):
        '''Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function computes all possible tokens for c (memory inefficient)'''
        prefix_length = len(g)
        last_char = g[-1] if prefix_length > 0 else 0

        # init. r
        r = np.full((self.input_length, 2, self.odim),
                    self.logzero, dtype=np.float32)

        # start from len(g) because is impossible for CTC to generate |y|>|X|
        start = max(1, prefix_length)

        if prefix_length == 0:
            r[0, 0, :] = self.x[0, :]    # if g = <sos>

        psi = r[start-1, 0, :]

        phi = np.logaddexp(r_prev[:, 0], r_prev[:, 1])

        for t in range(start, self.input_length):
            # prev_blank
            prev_blank = np.full((self.odim), r_prev[t-1, 1], dtype=np.float32)
            # prev_nonblank
            prev_nonblank = np.full(
                (self.odim), r_prev[t-1, 0], dtype=np.float32)
            prev_nonblank[last_char] = self.logzero

            phi = np.logaddexp(prev_nonblank, prev_blank)
            # P(h|current step is non-blank) = [ P(prev. step = y) + P()]*P(c)
            r[t, 0, :] = np.logaddexp(r[t-1, 0, :], phi) + self.x[t, :]
            # P(h|current step is blank) = [P(prev. step is blank) + P(prev. step is non-blank)]*P(now=blank)
            r[t, 1, :] = np.logaddexp(
                r[t-1, 1, :], r[t-1, 0, :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi+self.x[t, :])

        #psi[self.eos] = np.logaddexp(r_prev[-1,0], r_prev[-1,1])
        return psi, np.rollaxis(r, 2)

    def cheap_compute(self, g, r_prev, candidates):
        '''Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function considers only those tokens in candidates for c (memory efficient)'''
        prefix_length = len(g)
        odim = len(candidates)
        last_char = g[-1] if prefix_length > 0 else 0

        # init. r
        r = np.full((self.input_length, 2, len(candidates)),
                    self.logzero, dtype=np.float32)

        # start from len(g) because is impossible for CTC to generate |y|>|X|
        start = max(1, prefix_length)

        if prefix_length == 0:
            r[0, 0, :] = self.x[0, candidates]    # if g = <sos>

        psi = r[start-1, 0, :]
        # Phi = (prev_nonblank,prev_blank)
        sum_prev = np.logaddexp(r_prev[:, 0], r_prev[:, 1])
        phi = np.repeat(sum_prev[..., None],odim,axis=-1)
        # Handle edge case : last tok of prefix in candidates
        if  prefix_length>0 and last_char in candidates:
            phi[:,candidates.index(last_char)] = r_prev[:,1]

        for t in range(start, self.input_length):
            # prev_blank
            # prev_blank = np.full((odim), r_prev[t-1, 1], dtype=np.float32)
            # prev_nonblank
            # prev_nonblank = np.full((odim), r_prev[t-1, 0], dtype=np.float32)
            # phi = np.logaddexp(prev_nonblank, prev_blank)
            # P(h|current step is non-blank) =  P(prev. step = y)*P(c)
            r[t, 0, :] = np.logaddexp( r[t-1, 0, :], phi[t-1]) + self.x[t, candidates]
            # P(h|current step is blank) = [P(prev. step is blank) + P(prev. step is non-blank)]*P(now=blank)
            r[t, 1, :] = np.logaddexp( r[t-1, 1, :], r[t-1, 0, :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi[t-1,]+self.x[t, candidates])

        # P(end of sentence) = P(g)
        if self.eos in candidates:
            psi[candidates.index(self.eos)] = sum_prev[-1]
        return psi, np.rollaxis(r, 2)

class CTCHypothesis():
    ''' 
        Hypothesis for pure CTC beam search decoding.
        An implementation of Algo. 1 in http://proceedings.mlr.press/v32/graves14.pdf
    '''
    def __init__(self):
        self.y = []
        # All probabilities are computed in log scale
        self.Pr_y_t_blank         = 0.0      # Pr-(y,t-1)  -> Pr-(y,t)
        self.Pr_y_t_nblank        = LOG_ZERO # Pr+(y,t-1)  -> Pr+(y,t)

        self.Pr_y_t_blank_bkup    = 0.0      # Pr-(y,t-1)  -> Pr-(y,t)
        self.Pr_y_t_nblank_bkup   = LOG_ZERO # Pr+(y,t-1)  -> Pr+(y,t)
        
        self.lm_output = None
        self.lm_hidden = None
        self.updated_lm = False
    
    def update_lm(self, output, hidden):
        self.lm_output = output
        self.lm_hidden = hidden
        self.updated_lm = True

    def get_len(self):
        return len(self.y)
    
    def get_string(self):
        # Convert the output sequence from list to string
        return ''.join([str(s) for s in self.y])
    
    def get_score(self):
        return np.logaddexp(self.Pr_y_t_blank, self.Pr_y_t_nblank)
    
    def get_final_score(self):
        if len(self.y) > 0:
            return np.logaddexp(self.Pr_y_t_blank, self.Pr_y_t_nblank) / len(self.y)
        else:
            return np.logaddexp(self.Pr_y_t_blank, self.Pr_y_t_nblank)

    def check_same(self, y_2):
        if len(self.y) != len(y_2):
            return False
        for i in range(len(self.y)):
            if self.y[i] != y_2[i]:
                return False
        return True

    def update_Pr_nblank(self, ctc_y_t):
        # ctc_y_t  : Pr(ye,t|x)
        # Pr+(y,t) = Pr+(y,t-1) * Pr(ye,t|x)
        self.Pr_y_t_nblank += ctc_y_t

    def update_Pr_nblank_prefix(self, ctc_y_t, Pr_y_t_blank_prefix, Pr_y_t_nblank_prefix, Pr_ye_y=None):
        # ctc_y_t  : Pr(ye,t|x)
        lm_prob = Pr_ye_y if Pr_ye_y is not None else 0.0
        if len(self.y) == 0: return
        if len(self.y) == 1:
            Pr_ye_y_prefix = ctc_y_t + lm_prob + np.logaddexp(Pr_y_t_blank_prefix, Pr_y_t_nblank_prefix)
        else:
            # Pr_ye_y : LM Pr(ye|y)
            Pr_ye_y_prefix = ctc_y_t + lm_prob + (Pr_y_t_blank_prefix if self.y[-1] == self.y[-2] \
                                        else np.logaddexp(Pr_y_t_blank_prefix, Pr_y_t_nblank_prefix))
        # Pr+(y,t) = Pr+(y,t) + Pr(ye,y^,t)
        self.Pr_y_t_nblank = np.logaddexp(self.Pr_y_t_nblank, Pr_ye_y_prefix)
    
    def update_Pr_blank(self, ctc_blank_t):
        # Pr-(y,t) = Pr(y,t-1) * Pr(-,t|x)
        self.Pr_y_t_blank = np.logaddexp(self.Pr_y_t_nblank_bkup, self.Pr_y_t_blank_bkup) + ctc_blank_t
    
    def add_token(self, token, ctc_token_t, Pr_k_y=None):
        # Add token to the end of the sequence
        # Update current sequence probability
        lm_prob = Pr_k_y if Pr_k_y is not None else 0.0
        if len(self.y) == 0:
            Pr_y_t_nblank_new = ctc_token_t + lm_prob + np.logaddexp(self.Pr_y_t_blank_bkup, self.Pr_y_t_nblank_bkup)
        else:
            # Pr_k_y : LM Pr(k|y)
            Pr_y_t_nblank_new = ctc_token_t + lm_prob + (self.Pr_y_t_blank_bkup if self.y[-1] == token else \
                                    np.logaddexp(self.Pr_y_t_blank_bkup, self.Pr_y_t_nblank_bkup))

        self.Pr_y_t_blank  = LOG_ZERO
        self.Pr_y_t_nblank = Pr_y_t_nblank_new

        self.Pr_y_t_blank_bkup  = self.Pr_y_t_blank
        self.Pr_y_t_nblank_bkup = self.Pr_y_t_nblank

        self.y.append(token)

    def orig_backup(self):
        self.Pr_y_t_blank_bkup  = self.Pr_y_t_blank
        self.Pr_y_t_nblank_bkup = self.Pr_y_t_nblank

class CTCBeamDecoder(nn.Module):
    ''' Beam decoder for ASR (CTC only) '''
    def __init__(self, asr, vocab_range, beam_size, vocab_candidate,
            lm_path='', lm_config='', lm_weight=0.0, device=None):
        super().__init__()
        # Setup
        self.asr         = asr
        self.vocab_range = vocab_range
        self.beam_size   = beam_size
        self.vocab_cand  = vocab_candidate
        assert self.vocab_cand <= len(self.vocab_range)

        assert self.asr.enable_ctc

        # Setup RNNLM
        self.apply_lm = lm_weight > 0
        self.lm_w = 0
        if self.apply_lm:
            self.device = device
            self.lm_w = lm_weight
            self.lm_path = lm_path
            lm_config = yaml.load(open(lm_config, 'r'), Loader=yaml.FullLoader)
            self.lm = RNNLM(self.asr.vocab_size, **lm_config['model']).to(self.device)
            self.lm.load_state_dict(torch.load(
                self.lm_path, map_location='cpu')['model'])
            self.lm.eval()

    def create_msg(self):
        msg = ['Decode spec| CTC decoding \t| Beam size = {} \t| LM weight = {}'.format(self.beam_size, self.lm_w)]
        return msg

    def forward(self, feat, feat_len):
        # Init.
        assert feat.shape[0] == 1, "Batchsize == 1 is required for beam search"
        
        # Calculate CTC output probability
        ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.asr(feat, feat_len, 10)
        del encode_len, att_output, att_align, dec_state, feat_len
        ctc_output = F.log_softmax(ctc_output[0], dim=-1).cpu().detach().numpy()
        T = len(ctc_output) # ctc_output = Pr(k,t|x) / dim: T x Vocab

        # Best W probable sequences
        B = [CTCHypothesis()]
        if self.apply_lm:
            # 0 == <sos> for RNNLM
            output, hidden = \
                self.lm(torch.zeros((1,1),dtype=torch.long).to(self.device), torch.ones(1,dtype=torch.long).to(self.device), None)
            B[0].update_lm(
                (output).log_softmax(dim=-1).squeeze().cpu().numpy(),
                hidden
            )
        
        start = True
        for t in range(T):
            # greedily ignoring pads at the beginning of the sequence
            if np.argmax(ctc_output[t]) == 0 and start:
                continue
            else:
                start = False
            B_new = []
            for i in range(len(B)): # For y in B
                B_i_new = copy.deepcopy(B[i])
                if B_i_new.get_len() > 0: # If y is not empty
                    if B_i_new.y[-1] == 1:
                        # <eos> = 1 (reached the end)
                        B_new.append(B_i_new)
                        continue
                    B_i_new.update_Pr_nblank(ctc_output[t, B_i_new.y[-1]])
                    # Find the same prefix
                    for j in range(len(B)):
                        if i != j and B[j].check_same(B_i_new.y[:-1]):
                            lm_prob = 0.0
                            if self.apply_lm:
                                lm_prob = self.lm_w * B[j].lm_output[B_i_new.y[-1]]
                            B_i_new.update_Pr_nblank_prefix(ctc_output[t, B_i_new.y[-1]], 
                                B[j].Pr_y_t_blank, B[j].Pr_y_t_nblank, lm_prob)
                            break
                B_i_new.update_Pr_blank(ctc_output[t, 0]) # 0 == <pad>
                if self.apply_lm:
                    lm_hidden = B_i_new.lm_hidden
                    lm_probs = B_i_new.lm_output
                else:
                    lm_hidden = None
                    lm_probs = None
                
                # Sort the next possible output symbol by CTC (and LM) score
                if self.apply_lm:
                    ctc_vocab_cand = sorted(zip(
                        self.vocab_range, ctc_output[t, self.vocab_range] + self.lm_w * lm_probs[self.vocab_range]), 
                        reverse=True, key=lambda x: x[1])
                else:
                    ctc_vocab_cand = sorted(zip(self.vocab_range, ctc_output[t, self.vocab_range]), reverse=True, key=lambda x: x[1])
                # Select top K possible symbols to calculate the probabilities
                for j in range(self.vocab_cand):
                    # <pad>=0, <eos>=1, <unk>=2
                    k = ctc_vocab_cand[j][0]
                    # Pr(k,t|x)
                    hyp_yk = copy.deepcopy(B_i_new)
                    lm_prob = 0.0 if not self.apply_lm else self.lm_w * lm_probs[k]
                    hyp_yk.add_token(k, ctc_output[t, k], lm_prob)
                    hyp_yk.updated_lm = False
                    B_new.append(hyp_yk)
                B_i_new.orig_backup() # Retrieve origin prob. before add_token()
                B_new.append(B_i_new)
            del B
            B = []

            # Remove duplicated sequences by sorting first (O(NlogN))
            B_new = sorted(B_new, key=lambda x: x.get_string())
            B.append(B_new[0]) # First Hyp always unique
            for i in range(1,len(B_new)):
                if B_new[i].check_same(B[-1].y):
                    # Next Hyp is duplicated, pick the higher one
                    if B_new[i].get_score() > B[-1].get_score():
                        B[-1] = B_new[i]
                    continue
                else:
                    # Next Hyp is different, hence valid
                    B.append(B_new[i])
            del B_new

            # Find top W possible sequences
            if t == T - 1:
                B = sorted(B, reverse=True, key=lambda x: x.get_final_score())
            else:
                B = sorted(B, reverse=True, key=lambda x: x.get_score())
            if len(B) > self.beam_size:
                B = B[:self.beam_size]
            
            # Update LM states
            if self.apply_lm and t < T - 1:
                for i in range(len(B)):
                    if B[i].get_len() > 0 and not B[i].updated_lm:
                        output, hidden = \
                            self.lm(B[i].y[-1] * torch.ones((1,1), dtype=torch.long).to(self.device), 
                                torch.ones(1,dtype=torch.long).to(self.device), B[i].lm_hidden)
                        B[i].update_lm(
                            (output).log_softmax(dim=-1).squeeze().cpu().numpy(),
                            hidden
                        )
        
        return [b.y for b in B]
