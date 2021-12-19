"""
Audio Losses

-- Experiments with creating an STFT loss -- will probably just use auraloss
"""

import auraloss
import torch
import torch.nn as nn
from torchaudio.functional import detect_pitch_frequency


class PitchMultiSTFTLoss(nn.Module):
    """
    Combines pitch detection and multi-spectral loss
    """

    def __init__(self):
        super(PitchMultiSTFTLoss, self).__init__()
        self.multi_spectral = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, x, y):
        assert len(x.shape) == 3
        assert len(y.shape) == 3
        return self.multi_spectral(x, y)


class PitchLoss(nn.Module):
    """
    Loss on pitch
    """

    def __init__(self, sample_rate):
        super(PitchLoss, self).__init__()
        self.sample_rate = sample_rate
        self.window = torch.hann_window(1024)

    def stft(self, audio):
        self.window = self.window.to(audio.device)
        audio = audio.squeeze(1)
        x_stft = torch.stft(audio, n_fft=1024, hop_length=512, window=self.window, return_complex=True)
        x_mag = torch.sqrt(torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=1e-8))
        return torch.log10(x_mag)

    def forward(self, x, y):
        assert len(x.shape) == 3
        assert len(y.shape) == 3
        pitch_x = self.stft(x)
        print(pitch_x.shape)
        return 0.0
