import torch
from functools import partial
from src.text import load_text_encoder
from src.audio import create_transform
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

DEV_N_JOBS = 1           # Number of threads used for dev set
HALF_BATCHSIZE_LEN = 800 # Batch size will be halfed if the longest wavefile surpasses threshold
# Note: Bucketing may cause random sampling to be biased (less sampled for those length > HALF_BATCHSIZE_LEN ) 

def collect_audio_batch(batch, audio_transform, mode):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...] '''

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if len(batch) == 1:
        batch = batch[0]
    # Make sure that batch size is reasonable
    first_len = audio_transform(str(batch[0][0])).shape[0]
    if first_len > HALF_BATCHSIZE_LEN and mode=='train':
        batch = batch[:len(batch)//2]
    # Read batch
    audio_feat, audio_len, text = [],[],[]
    with torch.no_grad():
        for b in batch:
            feat = audio_transform(str(b[0]))
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(torch.LongTensor(b[1]))
    # Descending length
    audio_len, audio_feat, text = zip(*[(feat_len,feat,txt) \
                 for feat_len,feat,txt in sorted(zip(audio_len,audio_feat,text), reverse=True, key=lambda x:x[0])]) 
    # Zero-padding
    audio_feat = pad_sequence(audio_feat, batch_first=True)
    text = pad_sequence(text, batch_first=True)
    audio_len = torch.LongTensor(audio_len)
    
    return audio_feat, audio_len, text

def collect_text_batch(batch):
    '''Collects a batch of text, should be list of list of int token 
       e.g. [txt1 <list>,txt2 <list>,...] '''

    # Bucketed batch should be [[txt1, txt2,...]]
    if len(batch) == 1:
        batch = batch[0]
    # Read batch
    text = [torch.LongTensor(b) for b in batch]
    # Zero-padding
    text = pad_sequence(text, batch_first=True)
    
    return text


def create_dataset(tokenizer, train_split, dev_split, name, path, bucketing, batch_size):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    bucket_size = batch_size if bucketing else 1
    tr_loader_bs = 1 if bucketing else batch_size
    dv_set = Dataset(path,dev_split,tokenizer, 1) # Do not use bucketing for dev set
    tr_set = Dataset(path,train_split,tokenizer, bucket_size)

    # Messages to show
    msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                         dev_split.__str__(),len(dv_set),batch_size,bucketing)

    return tr_set, dv_set, tr_loader_bs, batch_size, msg_list

def create_textset(tokenizer, train_split, dev_split, name, path, bucketing, batch_size):
    ''' Interface for creating all kinds of text dataset'''
    msg_list = []

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriTextDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    bucket_size = batch_size if bucketing else 1
    tr_loader_bs = 1 if bucketing else batch_size
    dv_set = Dataset(path,dev_split,tokenizer, 1) # Do not use bucketing for dev set
    tr_set = Dataset(path,train_split,tokenizer, bucket_size)
    
    # Messages to show
    msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                         dev_split.__str__(),len(dv_set),batch_size,bucketing)

    return tr_set, dv_set, tr_loader_bs, batch_size, msg_list

def load_dataset(n_jobs, use_gpu, pin_memory, corpus, audio, text):
    ''' Prepare dataloader for training/validation'''

    # Audio feature extractor
    audio_transform, feat_dim = create_transform(audio.copy())
    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, data_msg = create_dataset(tokenizer,**corpus)
    # Dataloader
    collect_tr = partial(collect_audio_batch, audio_transform=audio_transform, mode='train')
    collect_dv = partial(collect_audio_batch, audio_transform=audio_transform, mode='dev')
    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=True, drop_last=True, collate_fn=collect_tr,
                        num_workers=max(0,n_jobs-DEV_N_JOBS), pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=DEV_N_JOBS, pin_memory=pin_memory)
    # Messages to show
    data_msg.append('I/O spec.  | Audio feature = {}\t | feature dim = {}\t| Token type = {}\t| Vocab size = {}'\
                    .format(audio['feat_type'],feat_dim,tokenizer.token_type,tokenizer.vocab_size))

    return tr_set, dv_set, feat_dim, tokenizer.vocab_size, tokenizer, data_msg


def load_textset(n_jobs, use_gpu, pin_memory, corpus, text):

    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, data_msg = create_textset(tokenizer,**corpus)
    # Dataloader
    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=True, drop_last=True, collate_fn=collect_text_batch,
                        num_workers=max(0,n_jobs-DEV_N_JOBS), pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_text_batch,
                        num_workers=DEV_N_JOBS, pin_memory=pin_memory)

    # Messages to show
    data_msg.append('I/O spec.  | Token type = {}\t| Vocab size = {}'\
                    .format(tokenizer.token_type,tokenizer.vocab_size))

    return tr_set, dv_set, tokenizer.vocab_size, tokenizer, data_msg


def _data_msg(name,path,train_split,tr_set,dev_split,dv_set,batch_size,bucketing):
    ''' List msg for verbose function '''
    msg_list = []
    msg_list.append('Data spec. | Corpus = {} (from {})'.format(name,path))
    msg_list.append('           | Train sets = {}\t| Number of utts = {}'.format(train_split,tr_set))
    msg_list.append('           | Dev sets = {}\t| Number of utts = {}'.format(dev_split,dv_set))
    msg_list.append('           | Batch size = {}\t| Bucketing = {}'.format(batch_size,bucketing))
    return msg_list
