from pathlib import Path
from os.path import join,getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset


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
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # List all wave files
        file_list = []
        for s in split:
            file_list += list(Path(join(path,s)).rglob("*.flac"))
        
        # Read text
        text = Parallel(n_jobs=-1)(delayed(read_text)(str(f)) for f in file_list)
        text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        
        # Read file size and sort dataset by file size (Note: feature len. may be different)
        file_len = Parallel(n_jobs=-1)(delayed(getsize)(f) for f in file_list)
        self.file_list, self.text = zip(*[(f_name,txt) \
                    for _,f_name,txt in sorted(zip(file_len,file_list,text), reverse=True, key=lambda x:x[0])])

    def __getitem__(self,index):
        if self.bucket_size>1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size,index)
            return [(f_path, txt) for f_path,txt in \
                     zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]


    def __len__(self):
        return len(self.file_list)
        


