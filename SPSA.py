# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
from torch.autograd import gradcheck
from torchsynth.config import SynthConfig
from torchsynth.synth import AbstractSynth, Voice
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt

BATCH_SIZE = 1
SR = 44100

# Load target
target, _ = librosa.load("./audio/snaredrum.wav", sr=SR)
ipd.Audio(target, rate=SR)

# Configure Synth
synth_config = SynthConfig(
    batch_size=BATCH_SIZE,
    buffer_size_seconds=len(target) / SR,
    sample_rate=SR,
    reproducible=False,
)
voice = Voice(synth_config)
# Run on the GPU if it's available
if torch.cuda.is_available():
    voice = voice.to("cuda")


class TorchSynthSPSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, synth):
        ctx.save_for_backward(input)
        ctx.synth = synth

        parameters = [param for _, param in sorted(synth.named_parameters())]
        for i, parameter in enumerate(parameters):
            parameter.requires_grad = False
            parameter.data = input[:, i]

        output, _, _ = synth()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        print("here in backward")
        return None


# +
synth_func = TorchSynthSPSA.apply

in_params = torch.rand(BATCH_SIZE, 78, requires_grad=True).to("cuda")
audio = synth_func(in_params, voice)
print(audio)
ipd.Audio(audio[0].detach().cpu().numpy(), rate=SR)


# -


class TorchSynth(nn.Module):
    """
    Module that contains randomly initialized synth paramaters
    """

    def __init__(self, synth: AbstractSynth):
        super(TorchSynth, self).__init__()
        self.synth = synth
        self.num_parameters = len(list(synth.parameters()))
        self.batch_size = synth.batch_size

        # These are parameters for the synth
        self.synth_parameters = nn.Parameter(
            torch.empty(self.batch_size, self.num_parameters)
        )
        nn.init.uniform_(self.synth_parameters)

    def forward(self):
        """
        Input is a set of parameters for the synth
        """
        return TorchSynthSPSA.apply(self.synth_parameters, self.synth)


synth = TorchSynth(voice).to("cuda")

audio = synth()
ipd.Audio(audio[0].detach().cpu().numpy(), rate=SR)

# +
## Try optimizing to find parameters for a target

# +
# STFT function
window = torch.hann_window(1024).to("cuda")


def stft(audio):
    x_stft = torch.stft(audio, n_fft=1024, window=window, return_complex=True)
    x_mag = torch.sqrt(torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=1e-8))
    return torch.log10(x_mag)


# Spectral loss A
target_spectrogram = stft(torch.tensor(target).to("cuda"))


def stft_loss(audio):
    spectrogram = stft(audio)
    return torch.mean(torch.square(spectrogram - target_spectrogram), dim=(1, 2))


# -

plt.imshow(target_spectrogram.cpu().numpy(), aspect="auto", origin="lower")

output_spec = stft(audio)
plt.imshow(output_spec[0].detach().cpu().numpy(), aspect="auto", origin="lower")

error = stft_loss(audio)
error.backward()
