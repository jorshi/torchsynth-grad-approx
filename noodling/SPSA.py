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

# +
import auraloss
import torch
import torch.nn as nn
from torch.autograd import gradcheck
from torchsynth.config import SynthConfig
from torchsynth.synth import AbstractSynth, Voice
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

from module import TorchSynthSPSA, TorchSynthModule

# %load_ext autoreload
# %autoreload 2
# -

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

# +
synth_func = TorchSynthSPSA.apply

in_params = torch.rand(BATCH_SIZE, 78, requires_grad=True).to("cuda")
audio = synth_func(in_params, voice)
print(audio)
ipd.Audio(audio[0].detach().cpu().numpy(), rate=SR)


# -


synth = TorchSynthModule(voice).to("cuda")
# print(list(synth.synth_parameters))

synth.randomize(seed=42)
audio = synth()
ipd.Audio(audio[0].detach().cpu().numpy(), rate=SR)

# +
mrstft = auraloss.freq.MultiResolutionSTFTLoss()


target_t = torch.tensor(target).to("cuda").expand(audio.shape[0], -1)[:, None, :]

loss = mrstft(target_t, audio[:, None, :])
print(loss)

# +
optimizer = torch.optim.Adam(synth.parameters(), lr=0.01)

pbar = tqdm(range(10000), desc="Iter 0")
for i in pbar:
    optimizer.zero_grad()

    audio = synth()
    error = mrstft(target_t, audio[:, None, :])
    pbar.set_description(f"Iter {i}: Error: {error}")

    error.backward()
    optimizer.step()
# -

plt.imshow(target_spectrogram.cpu().numpy(), aspect="auto", origin="lower")

output_spec = stft(audio)
plt.imshow(output_spec[0].detach().cpu().numpy(), aspect="auto", origin="lower")
