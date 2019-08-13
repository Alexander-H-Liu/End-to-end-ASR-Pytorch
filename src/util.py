import math
import time
import numpy as np
from torch import nn
import editdistance as ed


class Timer():
    ''' Timer for recording training time distribution. '''
    def __init__(self):
        self.prev_t = time()
        self.clear()

    def set(self):
        self.prev_t = time()

    def cnt(self,mode):
        self.time_table[mode] += time()-self.prev_t
        self.set()
        if mode =='bw':
            self.click += 1

    def show(self):
        total_time = sum(self.time_table.values())
        self.time_table['avg'] = total_time/self.click
        self.time_table['rd'] = 100*self.time_table['rd']/total_time
        self.time_table['fw'] = 100*self.time_table['fw']/total_time
        self.time_table['bw'] = 100*self.time_table['bw']/total_time
        msg  = '{avg:.3f} sec/step (rd {rd:.1f}% | fw {fw:.1f}% | bw {bw:.1f}%)'.format(self.time_table)
        self.clear()
        return msg

    def clear(self):
        self.time_table = {'rd':0,'fw':0,'bw':0}
        self.click = 0

# Reference : https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr.py#L168
def init_weights(module):
    # Exceptions
    if type(module) == nn.Embedding:
        module.weight.data.normal_(0, 1)
    else:
        for p in module.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in [3,4]:
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1. / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError

# Reference : https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
def human_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '{:3}{}'.format(num, [' ', 'K', 'M', 'G', 'T', 'P'][magnitude])

def cal_er(tokenizer, pred, truth, mode='wer'):
    # Calculate error rate of a batch
    if pred is None:
        return np.nan
    er = []
    for p,t in zip(pred,truth):
        p = tokenizer.decode(p.tolist())
        t = tokenizer.decode(t.tolist())
        if mode == 'wer':
            p = p.split(' ')
            t = t.split(' ')
        er.append(float(ed.eval(p,t))/len(t))
    return sum(er)/len(er)