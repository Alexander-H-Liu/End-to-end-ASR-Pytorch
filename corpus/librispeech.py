import torch
from pathlib import Path
from os.path import join,getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


HALF_BATCHSIZE_TIME=800 # when bucketing, batch size will be halfed if the longest wavefile surpasses this
# Note: Bucketing may cause random sampling to be biased (less sampled for those length > HALF_BATCHSIZE_TIME)

def read_text(file):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    src_file = '-'.join(file.split('-')[:-1])+'.trans.txt'
    idx = file.split('/')[-1].split('.')[0]

    with open(src_file,'r') as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ',1)[1]


class LibriDataset(Dataset):
    def __init__(self, path, split, tokenizer, audio_transform, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.transform = audio_transform

        # List all wave files
        self.file_list = []
        for s in split:
            self.file_list += list(Path(join(path,s)).rglob("*.flac"))
        
        # Read text
        self.text = Parallel(n_jobs=-1)(delayed(read_text)(str(f)) for f in self.file_list)
        self.text = Parallel(n_jobs=-1)(delayed(tokenizer.tokenize)(txt) for txt in self.text)
        
        if bucket_size>1:
            # Read file size and sort dataset by length
            file_len = Parallel(n_jobs=-1)(delayed(getsize)(str(f)) for f in self.file_list)
            self.file_list = [f_name for _,f_name in sorted(zip(file_len,self.file_list),reverse=True)]
            self.text = [txt for _,txt in sorted(zip(file_len,self.text),reverse=True)]

    def __getitem__(self,index):
        # load feature
        audio_feat = self.transform(self.file_list[index])
        
        if self.bucket_size>1:
            # Create bucket (half-sized if length >= HALF_BATCHSIZE_TIME)
            max_time = audio_feat.shape[0]
            audio_feats = [audio_feat]
            bucket_size = self.bucket_size//2 if max_time >= HALF_BATCHSIZE_TIME else self.bucket_size
            audio_feats += [self.transform(self.file_list[index+t]) for t in range(1,bucket_size)]
            audio_lens = [feat.shape[0] for feat in audio_feats]
            token_seqs  = [torch.LongTensor(txt) for txt in self.text[index:index+bucket_size]]
            # Zero-padding
            audio_feats = pad_sequence(audio_feats, batch_first=True)
            token_seqs = pad_sequence(token_seqs, batch_first=True)
            return audio_feats, audio_lens, token_seqs
        else:
            token_seq  = torch.LongTensor(self.text[index])
            return audio_feat, token_seq

    def __len__(self):
        return len(self.file_list)
        


