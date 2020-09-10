import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy

from src.lm import RNNLM
from src.ctc import CTCPrefixScore

CTC_BEAM_RATIO = 1.5   # DO NOT CHANGE THIS, MAY CAUSE OOM
LOG_ZERO = -10000000.0  # Log-zero for CTC


class BeamDecoder(nn.Module):
    ''' Beam decoder for ASR '''

    def __init__(self, asr, emb_decoder, beam_size, min_len_ratio, max_len_ratio,
                 lm_path='', lm_config='', lm_weight=0.0, ctc_weight=0.0):
        super().__init__()
        # Setup
        self.beam_size = beam_size
        self.min_len_ratio = min_len_ratio
        self.max_len_ratio = max_len_ratio
        self.asr = asr

        # ToDo : implement pure ctc decode
        assert self.asr.enable_att

        # Additional decoding modules
        self.apply_ctc = ctc_weight > 0
        if self.apply_ctc:
            assert self.asr.ctc_weight > 0, 'ASR was not trained with CTC decoder'
            self.ctc_w = ctc_weight
            self.ctc_beam_size = int(CTC_BEAM_RATIO * self.beam_size)

        self.apply_lm = lm_weight > 0
        if self.apply_lm:
            self.lm_w = lm_weight
            self.lm_path = lm_path
            lm_config = yaml.load(open(lm_config, 'r'), Loader=yaml.FullLoader)
            self.lm = RNNLM(self.asr.vocab_size, **lm_config['model'])
            self.lm.load_state_dict(torch.load(
                self.lm_path, map_location='cpu')['model'])
            self.lm.eval()

        self.apply_emb = emb_decoder is not None
        if self.apply_emb:
            self.emb_decoder = emb_decoder

    def create_msg(self):
        msg = ['Decode spec| Beam size = {}\t| Min/Max len ratio = {}/{}'.format(
            self.beam_size, self.min_len_ratio, self.max_len_ratio)]
        if self.apply_ctc:
            msg.append(
                '           |Joint CTC decoding enabled \t| weight = {:.2f}\t'.format(self.ctc_w))
        if self.apply_lm:
            msg.append('           |Joint LM decoding enabled \t| weight = {:.2f}\t| src = {}'.format(
                self.lm_w, self.lm_path))
        if self.apply_emb:
            msg.append('           |Joint Emb. decoding enabled \t| weight = {:.2f}'.format(
                self.lm_w, self.emb_decoder.fuse_lambda.mean().cpu().item()))

        return msg

    def forward(self, audio_feature, feature_len):
        # Init.
        assert audio_feature.shape[0] == 1, "Batchsize == 1 is required for beam search"
        batch_size = audio_feature.shape[0]
        device = audio_feature.device
        dec_state = self.asr.decoder.init_state(
            batch_size)                           # Init zero states
        self.asr.attention.reset_mem()            # Flush attention mem
        # Max output len set w/ hyper param.
        max_output_len = int(
            np.ceil(feature_len.cpu().item()*self.max_len_ratio))
        # Min output len set w/ hyper param.
        min_output_len = int(
            np.ceil(feature_len.cpu().item()*self.min_len_ratio))
        # Store attention map if location-aware
        store_att = self.asr.attention.mode == 'loc'
        prev_token = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device)     # Start w/ <sos>
        # Cache of beam search
        final_hypothesis, next_top_hypothesis = [], []
        # Incase ctc is disabled
        ctc_state, ctc_prob, candidates, lm_state = None, None, None, None

        # Encode
        encode_feature, encode_len = self.asr.encoder(
            audio_feature, feature_len)

        # CTC decoding
        if self.apply_ctc:
            ctc_output = F.log_softmax(
                self.asr.ctc_layer(encode_feature), dim=-1)
            ctc_prefix = CTCPrefixScore(ctc_output)
            ctc_state = ctc_prefix.init_state()

        # Start w/ empty hypothesis
        prev_top_hypothesis = [Hypothesis(decoder_state=dec_state, output_seq=[],
                                          output_scores=[], lm_state=None, ctc_prob=0,
                                          ctc_state=ctc_state, att_map=None)]
        # Attention decoding
        for t in range(max_output_len):
            for hypothesis in prev_top_hypothesis:
                # Resume previous step
                prev_token, prev_dec_state, prev_attn, prev_lm_state, prev_ctc_state = hypothesis.get_state(
                    device)
                self.asr.set_state(prev_dec_state, prev_attn)

                # Normal asr forward
                attn, context = self.asr.attention(
                    self.asr.decoder.get_query(), encode_feature, encode_len)
                asr_prev_token = self.asr.pre_embed(prev_token)
                decoder_input = torch.cat([asr_prev_token, context], dim=-1)
                cur_prob, d_state = self.asr.decoder(decoder_input)

                # Embedding fusion (output shape 1xV)
                if self.apply_emb:
                    _, cur_prob = self.emb_decoder( d_state, cur_prob, return_loss=False)
                else:
                    cur_prob = F.log_softmax(cur_prob, dim=-1)

                # Perform CTC prefix scoring on limited candidates (else OOM easily)
                if self.apply_ctc:
                    # TODO : Check the performance drop for computing part of candidates only
                    _, ctc_candidates = cur_prob.squeeze(0).topk(self.ctc_beam_size, dim=-1)
                    candidates = ctc_candidates.cpu().tolist()
                    ctc_prob, ctc_state = ctc_prefix.cheap_compute(
                        hypothesis.outIndex, prev_ctc_state, candidates)
                    # TODO : study why ctc_char (slightly) > 0 sometimes
                    ctc_char = torch.FloatTensor(ctc_prob - hypothesis.ctc_prob).to(device)

                    # Combine CTC score and Attention score (HACK: focus on candidates, block others)
                    hack_ctc_char = torch.zeros_like(cur_prob).data.fill_(LOG_ZERO)
                    for idx, char in enumerate(candidates):
                        hack_ctc_char[0, char] = ctc_char[idx]
                    cur_prob = (1-self.ctc_w)*cur_prob + self.ctc_w*hack_ctc_char  # ctc_char
                    cur_prob[0, 0] = LOG_ZERO  # Hack to ignore <sos>

                # Joint RNN-LM decoding
                if self.apply_lm:
                    # assuming batch size always 1, resulting 1x1
                    lm_input = prev_token.unsqueeze(1)
                    lm_output, lm_state = self.lm(
                        lm_input, torch.ones([batch_size]), hidden=prev_lm_state)
                    # assuming batch size always 1,  resulting 1xV
                    lm_output = lm_output.squeeze(0)
                    cur_prob += self.lm_w*lm_output.log_softmax(dim=-1)

                # Beam search
                # Note: Ignored batch dim.
                topv, topi = cur_prob.squeeze(0).topk(self.beam_size)
                prev_attn = self.asr.attention.att_layer.prev_att.cpu() if store_att else None
                final, top = hypothesis.addTopk(topi, topv, self.asr.decoder.get_state(), att_map=prev_attn,
                                                lm_state=lm_state, ctc_state=ctc_state, ctc_prob=ctc_prob,
                                                ctc_candidates=candidates)
                # Move complete hyps. out
                if final is not None and (t >= min_output_len):
                    final_hypothesis.append(final)
                    if self.beam_size == 1:
                        return final_hypothesis
                next_top_hypothesis.extend(top)

            # Sort for top N beams
            next_top_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)
            prev_top_hypothesis = next_top_hypothesis[:self.beam_size]
            next_top_hypothesis = []

        # Rescore all hyp (finished/unfinished)
        final_hypothesis += prev_top_hypothesis
        final_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)

        return final_hypothesis[:self.beam_size]


class Hypothesis:
    '''Hypothesis for beam search decoding.
       Stores the history of label sequence & score 
       Stores the previous decoder state, ctc state, ctc score, lm state and attention map (if necessary)'''

    def __init__(self, decoder_state, output_seq, output_scores, lm_state, ctc_state, ctc_prob, att_map):
        assert len(output_seq) == len(output_scores)
        # attention decoder
        self.decoder_state = decoder_state
        self.att_map = att_map

        # RNN language model
        if type(lm_state) is tuple:
            self.lm_state = (lm_state[0].cpu(),
                             lm_state[1].cpu())  # LSTM state
        elif lm_state is None:
            self.lm_state = None                                  # Init state
        else:
            self.lm_state = lm_state.cpu()                        # GRU state

        # Previous outputs
        self.output_seq = output_seq        # Prefix, List of list
        self.output_scores = output_scores  # Prefix score, list of float

        # CTC decoding
        self.ctc_state = ctc_state          # List of np
        self.ctc_prob = ctc_prob            # List of float

    def avgScore(self):
        '''Return the averaged log probability of hypothesis'''
        assert len(self.output_scores) != 0
        return sum(self.output_scores) / len(self.output_scores)

    def addTopk(self, topi, topv, decoder_state, att_map=None,
                lm_state=None, ctc_state=None, ctc_prob=0.0, ctc_candidates=[]):
        '''Expand current hypothesis with a given beam size'''
        new_hypothesis = []
        term_score = None
        ctc_s, ctc_p = None, None
        beam_size = topi.shape[-1]

        for i in range(beam_size):
            # Detect <eos>
            if topi[i].item() == 1:
                term_score = topv[i].cpu()
                continue

            idxes = self.output_seq[:]     # pass by value
            scores = self.output_scores[:]  # pass by value
            idxes.append(topi[i].cpu())
            scores.append(topv[i].cpu())
            if ctc_state is not None:
                # ToDo: Handle out-of-candidate case.
                idx = ctc_candidates.index(topi[i].item())
                ctc_s = ctc_state[idx, :, :]
                ctc_p = ctc_prob[idx]
            new_hypothesis.append(Hypothesis(decoder_state,
                                             output_seq=idxes, output_scores=scores, lm_state=lm_state,
                                             ctc_state=ctc_s, ctc_prob=ctc_p, att_map=att_map))
        if term_score is not None:
            self.output_seq.append(torch.tensor(1))
            self.output_scores.append(term_score)
            return self, new_hypothesis
        return None, new_hypothesis

    def get_state(self, device):
        prev_token = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        prev_token = torch.LongTensor([prev_token]).to(device)
        att_map = self.att_map.to(device) if self.att_map is not None else None
        if type(self.lm_state) is tuple:
            lm_state = (self.lm_state[0].to(device),
                        self.lm_state[1].to(device))  # LSTM state
        elif self.lm_state is None:
            lm_state = None                                  # Init state
        else:
            lm_state = self.lm_state.to(
                device)                        # GRU state
        return prev_token, self.decoder_state, att_map, lm_state, self.ctc_state

    @property
    def outIndex(self):
        return [i.item() for i in self.output_seq]

class CTCHypothesis():
    ''' Hypothesis for pure CTC beam search decoding. '''
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
    def __init__(self, asr, vocab_range, beam_size, vocab_candidate, max_len_ratio,
            lm_path='', lm_config='', lm_weight=0.0, lm_temperature=1.0, device=None):
        super().__init__()
        # Setup
        self.asr         = asr
        self.vocab_range = vocab_range
        self.beam_size   = beam_size
        self.vocab_cand  = vocab_candidate
        self.max_len_ratio = max_len_ratio
        assert self.vocab_cand <= len(self.vocab_range)

        assert self.asr.enable_ctc

        self.apply_lm = lm_weight > 0
        self.lm_w = 0
        if self.apply_lm:
            self.device = device
            self.lm_w = lm_weight
            self.lm_temp = lm_temperature
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
        
        ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.asr(feat, feat_len, 10)
        del encode_len, att_output, att_align, dec_state, feat_len
        
        ctc_output = F.log_softmax(ctc_output[0], dim=-1).cpu().detach().numpy()
        greedy_seq = np.argmax(ctc_output, axis=-1).tolist()
        T = len(ctc_output) # ctc_output = Pr(k,t|x) / dim: T x Vocab

        # Best W probable sequences
        B = [CTCHypothesis()]
        if self.apply_lm:
            # 0 == <sos> for RNNLM
            output, hidden = \
                self.lm(torch.zeros((1,1),dtype=torch.long).to(self.device), torch.ones(1,dtype=torch.long).to(self.device), None)
            B[0].update_lm(
                (output / self.lm_temp).log_softmax(dim=-1).squeeze().cpu().numpy(),
                hidden
            )
        
        start = True
        for t in range(T):
            # greedily ignoring pads at the beginning of the sequence
            if np.argmax(ctc_output[t]) == 0 and start: continue
            else: start = False
            B_new = []
            for i in range(len(B)): # For y in B
                B_i_new = copy.deepcopy(B[i])
                if B_i_new.get_len() > 0: # If y is not empty
                    if B_i_new.y[-1] == 1:
                        # <eos> = 1 (reached the end)
                        B_new.append(B_i_new)
                        continue
                    B_i_new.update_Pr_nblank(ctc_output[t, B_i_new.y[-1]])
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
                if self.apply_lm:
                    ctc_vocab_cand = sorted(zip(
                        self.vocab_range, ctc_output[t,self.vocab_range] + self.lm_w * lm_probs[self.vocab_range]), 
                        reverse=True, key=lambda x: x[1])
                else:
                    ctc_vocab_cand = sorted(zip(self.vocab_range, ctc_output[t,self.vocab_range]), reverse=True, key=lambda x: x[1])
                for j in range(self.vocab_cand):
                    # <pad>=0, <eos>=1, <unk>=2
                    k = ctc_vocab_cand[j][0]
                    # Pr(k,t|x)
                    hyp_yk = copy.deepcopy(B_i_new)
                    lm_prob = 0.0 if not self.apply_lm else self.lm_w * lm_probs[k]
                    hyp_yk.add_token(k, ctc_output[t, k], lm_prob)
                    hyp_yk.updated_lm = False
                    B_new.append(hyp_yk)
                B_i_new.orig_backup()
                B_new.append(B_i_new)
            del B
            B = []
            B_new = sorted(B_new, key=lambda x: x.get_string())
            # Remove duplicated sequences
            for i in range(len(B_new)):
                if len(B) > 0:
                    if B_new[i].check_same(B[-1].y):
                        if B_new[i].get_score() > B[-1].get_score():
                            B[-1] = B_new[i]
                        continue
                    else: B.append(B_new[i])
                else: B.append(B_new[i])
            del B_new
            if t == T - 1:
                B = sorted(B, reverse=True, key=lambda x: x.get_final_score())
            else:
                B = sorted(B, reverse=True, key=lambda x: x.get_score())
            if len(B) > self.beam_size:
                B = B[:self.beam_size]
            # del B_new
            if self.apply_lm and t < T - 1:
                for i in range(len(B)):
                    if B[i].get_len() > 0 and not B[i].updated_lm:
                        output, hidden = \
                            self.lm(B[i].y[-1] * torch.ones((1,1), dtype=torch.long).to(self.device), 
                                torch.ones(1,dtype=torch.long).to(self.device), B[i].lm_hidden)
                        B[i].update_lm(
                            (output / self.lm_temp).log_softmax(dim=-1).squeeze().cpu().numpy(),
                            hidden
                        )
        
        return [b.y for b in B] + [greedy_seq]
