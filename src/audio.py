import torch.nn as nn
import torchaudio


class CMVN(nn.Module):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=1, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, time]
        # so perform normalization on dim=1 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


# TODO(WindQAQ): find more elegant way to deal with one line forward
class ReadAudio(nn.Module):
    def forward(self, filepath):
        return torchaudio.load(filepath)[0]


class Postprocess(nn.Module):
    def forward(self, x):
        return x[0].transpose(1, 0).detach()


def create_transform(audio_config):
    feat_type = audio_config.pop("feat_type", "fbank")
    feat_dim = audio_config.pop("feat_dim")
    apply_cmvn = audio_config.pop("apply_cmvn")

    transforms = [ReadAudio()]

    if feat_type == "fbank":
        transforms.append(torchaudio.transforms.MelSpectrogram(
            n_mels=feat_dim, **audio_config))
    elif feat_type == "mfcc":
        transforms.append(torchaudio.transforms.MFCC(
            n_mfcc=feat_dim, melkwargs=audio_config))
    else:
        raise NotImplementedError("Only support `fbank` and `mfcc`.")

    if apply_cmvn:
        transforms.append(CMVN())

    transforms.append(Postprocess())

    return nn.Sequential(*transforms)
