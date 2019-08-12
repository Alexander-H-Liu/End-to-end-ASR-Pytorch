from src.text import create_tokenizer
from src.audio import create_transform
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

DEV_N_JOBS = 2 # Number of threads used for dev set

def collect_batch(batch):
    '''Collects a batch, check if it's a bucket or not'''
    if len(batch) == 1:
        # Bucket
        return batch[0][0],batch[0][1],batch[0][2]
    else:
        # Normal batch
        audio_feat, audio_len, text = [],[],[]
        for b in batch:
            audio_feat.append(b[0])
            audio_len.append(b[0].shape[0])
            text.append(b[1])
        # Descending length
        audio_feat = [x for _,x in sorted(zip(audio_len,audio_feat),reverse=True)]
        text = [x for _,x in sorted(zip(audio_len,text),reverse=True)]
        audio_len = sorted(audio_len,reverse=True)
        # Zero-padding
        audio_feat = pad_sequence(audio_feat, batch_first=True)
        text = pad_sequence(text, batch_first=True)
        return audio_feat, audio_len, text


def create_dataset(tokenizer, audio_transform, name, path, bucket_size, bucketing, batch_size):
    msg_list = []

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    bucket_size = batch_size if bucketing else 1
    actual_batch_size = 1 if bucketing else batch_size
    tr_set = Dataset(path,train_split,tokenizer,audio_transform, bucket_size)
    dv_set = Dataset(path,dev_split,tokenizer,audio_transform, bucket_size)

    # Messages to show
    msg_list.append('Data spec. | Corpus = {} (from {})'.format(name,path))
    msg_list.append('           | Train sets = {}\t| # of utts = {}'.format(train_split.__str__(),len(tr_set)))
    msg_list.append('           | Dev sets = {}\t| # of utts = {}'.format(dev_split.__str__(),len(dv_set)))
    msg_list.append('           | Batch size = {}\t| Bucketing = {}'.format(batch_size,bucketing))

    return tr_set, dv_set, actual_batch_size, msg_list

def load_dataset(n_jobs, use_gpu, corpus, audio, text):
    msg_list = []

    # Audio feature extractor
    audio_transform, feat_dim = create_transform(apply_jit,**audio)
    # Text tokenizer
    tokenizer = create_tokenizer(**text)
    # Dataset
    tr_set, dv_set, batch_size, data_msg = create_dataset(tokenizer,audio_transform,**corpus)
    # Dataloader
    tr_set = DataLoader(tr_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collect_batch,
                        num_workers=max(0,n_jobs-DEV_N_JOBS), pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collect_batch,
                        num_workers=DEV_N_JOBS, pin_memory=use_gpu)


    # Messages to show
    msg_list.extend(data_msg)
    msg_list.append('I/O spec. | Audio feature = {}\t | feature dim = {}\t| Token type = {}\t| Vocab size = {}'\
                    .format(audio['feat_type'],feat_dim,tokenizer.tok_type,tokenizer.vocab_size))

    return tr_set, dv_set, feat_dim, tokenizer.vocab_size, tokenizer, msg_list