import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from src.util import init_weights 
from src.module import VGGExtractor, RNNLayer, ScaleDotAttention, LocationAwareAttention

class ASR(nn.Module):
    ''' ASR model, including Encoder/Decoder(s)'''
    def __init__(self, input_size, vocab_size, ctc_weight, encoder, attention, decoder):
        super(ASR, self).__init__()

        # Setup
        assert 0<=ctc_weight<=1
        self.ctc_weight = ctc_weight
        self.enable_ctc = ctc_weight > 0
        self.enable_att = ctc_weight != 1
        self.lm = None

        # Modules
        self.encoder = Encoder(input_size, **encoder)
        if self.enable_ctc:
            self.ctc_layer = nn.Linear(self.encoder.out_dim, vocab_size)
        if self.enable_att:
            self.dec_dim = decoder['dim']
            self.pre_embed = nn.Embedding(vocab_size, self.dec_dim)
            self.decoder = Decoder(self.encoder.out_dim+self.dec_dim, vocab_size, **decoder)
            self.attention = Attention(self.encoder.out_dim, self.dec_dim, **attention)

        self.apply(init_weights)

    def load_lm(self):
        pass #ToDo

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(self.encoder.sample_rate))
        if self.encoder.vgg:
            msg.append('           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
        if self.enable_ctc:
            msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(self.ctc_weight))
        if self.enable_att:
            msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(self.attention.mode,1-self.ctc_weight))
        return msg

    def forward(self, audio_feature, feature_len, decode_step,tf_rate=0.0,teacher=None):
        # Init
        bs = audio_feature.shape[0]
        ctc_output = None
        att_output = None
        att_seq = None

        # Encode
        encode_feature,encode_len = self.encoder(audio_feature,feature_len)

        # CTC based decoding
        if self.enable_ctc:
            ctc_output = F.log_softmax(self.ctc_layer(encode_feature),dim=-1)

        # Attention based decoding
        if self.enable_att:
            # Init (init char = <SOS>, reset all rnn state and cell)
            self.decoder.init_state(encode_feature)
            self.attention.reset_mem()
            last_char = self.pre_embed(torch.zeros((bs),dtype=torch.long, device=encode_feature.device))
            output_seq = []
            att_seq = []

            # Preprocess data for teacher forcing
            if teacher is not None:
                teacher = self.pre_embed(teacher)

            # Decode
            for t in range(decode_step):
                # Attend (inputs current state of first layer, encoded features)
                attn,context = self.attention(self.decoder.get_state(),encode_feature,encode_len)
                # Decode (inputs context + embedded last character)                
                decoder_input = torch.cat([last_char,context],dim=-1)
                cur_char = self.decoder(decoder_input)
                # Prepare output as input of next step
                if (teacher is not None):
                    if random.random() <= tf_rate:
                        # teacher forcing
                        last_char = teacher[:,t,:]
                    else:
                        # self-sampling (replace by argmax may be another choice)
                        sampled_char = Categorical(F.softmax(cur_char,dim=-1)).sample()
                        last_char = self.pre_embed(sampled_char)
                else:
                    # argmax for inference
                    last_char = self.pre_embed(torch.argmax(cur_char,dim=-1))

                # save output of each step
                output_seq.append(cur_char)
                att_seq.append(attn)

            att_output = torch.stack(output_seq,dim=1) # BxTxV
            att_seq = torch.stack(att_seq,dim=2)       # BxNxDtxT

        return ctc_output, encode_len, att_output, att_seq



class Decoder(nn.Module):
    ''' Decoder (a.k.a. Speller in LAS) '''
    # ToDo:　More elegant way to implement decoder 
    def __init__(self, input_dim, vocab_size, module, dim, layer, dropout, layer_norm):
        super(Decoder, self).__init__()
        self.in_dim = input_dim
        self.layer = layer
        self.dim = dim
        self.dropout = dropout
        self.layer_norm = layer_norm

        # Init 
        self.module = module+'Cell'
        self.state_list = []
        self.enable_cell = False
        if module  == 'LSTM':
            self.enable_cell = True
            self.cell_list = []
        elif module not in ['LSTM','GRU']:
            raise NotImplementedError
        
        # Modules
        module_list = []
        in_dim = input_dim
        for i in range(layer):
            module_list.append(getattr(nn,self.module)(in_dim,dim))
            in_dim = dim
        
        # Regularization
        if self.layer_norm:
            self.ln_list = nn.ModuleList([nn.LayerNorm(dim) for l in range(layer)])
        if self.dropout > 0:
            self.dp = nn.Dropout(self.dropout)

        self.layers = nn.ModuleList(module_list)
        self.char_trans = nn.Linear(dim,vocab_size)

        
    def init_state(self, context):
        # Set all hidden states to zeros
        self.state_list = [torch.zeros((context.shape[0],self.dim),device=context.device)]*self.layer
        if self.enable_cell:
            self.cell_list = [torch.zeros((context.shape[0],self.dim),device=context.device)]*self.layer

    def get_state(self):
        return self.state_list[0]

    def _get_layer_state(self, layer_idx):
        # Get hidden state of specified layer
        if self.enable_cell:
            return (self.state_list[layer_idx],self.cell_list[layer_idx])
        else:
            return self.state_list[layer_idx]

    def _store_layer_state(self, layer_idx, state):
        # Replace hidden state of specified layer
        if self.enable_cell:
            self.state_list[layer_idx] = state[0]
            self.cell_list[layer_idx] = state[1]
            return state[0]
        else:
            self.state_list[layer_idx] = state
            return state

    def forward(self, x):
        for i, layers in enumerate(self.layers):
            state = self._get_layer_state(i)
            x = layers(x,state)
            x = self._store_layer_state(i,x)

            if self.layer_norm:
                x = self.ln_list[i](x)
            if self.dropout > 0:
                x = self.dp(x)

        x = self.char_trans(x)
        
        return x




class Attention(nn.Module):  
    ''' Attention mechanism
        please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
        Input : Decoder state                      with shape [batch size, decoder hidden dimension]
                Compressed feature from Encoder    with shape [batch size, T, encoder feature dimension]
        Output: Attention score                    with shape [batch size, num head, T (attention score of each time step)]
                Context vector                     with shape [batch size, encoder feature dimension]
                (i.e. weighted (by attention score) sum of all timesteps T's feature) '''
    def __init__(self, v_dim, q_dim, mode, dim, num_head, temperature,
                 loc_kernel_size, loc_kernel_num):
        super(Attention,self).__init__()

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.mode = mode.lower()
        self.num_head = num_head

        # Linear proj. before attention
        self.proj_q = nn.Linear( q_dim, dim*num_head)
        self.proj_k = nn.Linear( v_dim, dim*num_head)
        self.proj_v = nn.Linear( v_dim, v_dim*num_head)

        # Attention
        if self.mode == 'dot':
            self.att_layer = ScaleDotAttention(temperature, self.num_head)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(loc_kernel_size, loc_kernel_num, dim, num_head, temperature)
        else:
            raise NotImplementedError

        # Layer for merging MHA
        if self.num_head > 1:
            self.merge_head = nn.Linear(v_dim*num_head, v_dim)
        
        # Stored feature
        self.key = None
        self.value = None
        self.mask = None
    
    def reset_mem(self):
        self.key = None
        self.value = None
        self.mask = None
        self.att_layer.reset_mem()

    def forward(self, dec_state, enc_feat, enc_len):

        # Preprecessing
        bs,ts,_ = enc_feat.shape
        query =  torch.tanh(self.proj_q(dec_state))
        query = query.view(bs, self.num_head, self.dim).view(bs*self.num_head, self.dim) # BNxD

        if self.key is None:
            # Maskout attention score for padded states
            self.att_layer.compute_mask(enc_len.to(enc_feat.device))

            # Store enc state to lower computational cost
            self.key =  torch.tanh(self.proj_k(enc_feat))
            self.value = torch.tanh(self.proj_v(enc_feat))
            if self.num_head>1:
                self.key = self.key.view(bs,ts,self.num_head,self.dim).permute(0,2,1,3) # BxNxTxD
                self.key = self.key.contiguous().view(bs*self.num_head,ts,self.dim) # BNxTxD
                self.value = self.value.view(bs,ts,self.num_head,self.v_dim).permute(0,2,1,3) # BxNxTxD
                self.value = self.value.contiguous().view(bs*self.num_head,ts,self.v_dim) # BNxTxD


        # Calculate attention    
        context, attn = self.att_layer(query, self.key, self.value)
        if self.num_head>1:
            context = context.view(bs,self.num_head*self.v_dim)    # BNxD  -> BxND
            context = self.merge_head(context) # BxD
        
        return attn,context



class Encoder(nn.Module):
    ''' Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details.'''
    def __init__(self, input_size, vgg, module, bidirection, dim, dropout, layer_norm, sample_rate, sample_style):
        super(Encoder, self).__init__()

        # Hyper-parameters checking
        self.vgg = vgg
        self.sample_rate = 1
        assert len(sample_rate)==len(dropout), 'Number of layer mismatch'
        assert len(dropout)==len(dim), 'Number of layer mismatch'
        num_layers = len(dim)
        assert num_layers>=1,'Encoder should have at least 1 layer'

        # Construct model
        module_list = []
        input_dim = input_size

        if vgg:
            vgg_extractor = VGGExtractor(input_size)
            module_list.append(vgg_extractor)
            input_dim = vgg_extractor.out_dim
            self.sample_rate = self.sample_rate*4

        if module in ['LSTM','GRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l], bidirection, dropout[l], layer_norm[l],
                                            sample_rate[l], sample_style))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate*sample_rate[l]
        else:
            raise NotImplementedError

        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

    def forward(self,input_x,enc_len):
        for _, layer in enumerate(self.layers):
            input_x,enc_len = layer(input_x,enc_len)
        return input_x,enc_len
