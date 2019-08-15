import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNLM(nn.Module):
    ''' RNN Language Model '''
    def __init__(self, vocab_size, emb_tying, emb_dim, module, dim, n_layers, dropout):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.emb_tying = emb_tying
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.dropout = dropout>0
        if self.dropout:
            self.dp = nn.Dropout(dropout)
        self.rnn = getattr(nn, module.upper())(emb_dim, dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.post_rnn = nn.Linear(dim,emb_dim,bias=None)
        if not self.emb_tying:
            self.trans = nn.Linear(emb_dim,vocab_size,bias=None)

    def create_msg(self):
        # Messages for user
        msg = ['Model spec.| RNNLM weight tying = {}, # of layers = {}, dim = {}'.format(self.emb_tying,self.n_layers,self.dim)]
        return msg

    def forward(self, x, lens, hidden=None):
        emb_x = self.emb(x)
        if self.dropout:
            emb_x = self.dp(emb_x)

        packed = nn.utils.rnn.pack_padded_sequence(emb_x, lens,batch_first=True)
        outputs, hidden = self.rnn(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        outputs = self.post_rnn(outputs)
        if self.emb_tying:
            outputs = F.linear(outputs,self.emb.weight)
        else:
            outputs = self.trans(outputs)
        return outputs, hidden
