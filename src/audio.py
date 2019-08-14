import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class CMVN(nn.Module):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=1, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
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


class Delta(nn.Module):

    __constants__ = ["order", "window_size", "padding"]

    def __init__(self, order=1, window_size=2):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        filters = self._create_filters(order, window_size)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    def forward(self, x):
        # Unsqueeze batch dim
        x = x.unsqueeze(0)
        return F.conv2d(x, weight=self.filters, padding=self.padding)[0]

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)


# TODO(WindQAQ): find more elegant way to deal with one line forward
class ReadAudio(nn.Module):
    def forward(self, filepath):
        return torchaudio.load(filepath)[0]


class Postprocess(nn.Module):
    def forward(self, x):
        # [channel, feature_dim, time] -> [time, channel, feature_dim]
        x = x.transpose(0, 2).transpose(1, 2)
        # [time, channel, feature_dim] -> [time, feature_dim * channel]
        return x.reshape(x.size(0), -1).detach()


def create_transform(audio_config):
    feat_type = audio_config.pop("feat_type", "fbank")
    feat_dim = audio_config.pop("feat_dim")

    delta_order = audio_config.pop("delta_order", 0)
    delta_window_size = audio_config.pop("delta_window_size", 2)
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

    if delta_order >= 1:
        transforms.append(Delta(delta_order, delta_window_size))

    if apply_cmvn:
        transforms.append(CMVN())

    transforms.append(Postprocess())

    return nn.Sequential(*transforms), feat_dim * (delta_order + 1)
