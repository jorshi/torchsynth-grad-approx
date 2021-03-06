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
from synth import SimpleFMSynth
from torchsynth.config import SynthConfig
import torch
import IPython.display as ipd
from tqdm import tqdm

from module import TorchSynthModule

# %load_ext autoreload
# %autoreload 2

# +
BATCH_SIZE = 1
SR = 44100
DURATION = 1.0

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# -

# Configure Synth
synth_config = SynthConfig(
    batch_size=BATCH_SIZE,
    buffer_size_seconds=DURATION,
    sample_rate=SR,
    reproducible=False,
)
synth = SimpleFMSynth(synth_config)
synth = synth.to(device)

synth.get_parameters()

synth.unfreeze_all_parameters()
synth.set_parameters(
    {
        ("keyboard", "midi_f0"): torch.tensor([48] * BATCH_SIZE),
        ("keyboard", "duration"): torch.tensor([0.9] * BATCH_SIZE),
        ("adsr", "attack"): torch.tensor([0.01] * BATCH_SIZE),
        # ("adsr", "decay"): torch.tensor([0.5] * BATCH_SIZE),
        ("adsr", "sustain"): torch.tensor([0.0] * BATCH_SIZE),
        ("adsr", "release"): torch.tensor([0.1] * BATCH_SIZE),
        ("adsr", "alpha"): torch.tensor([5.0] * BATCH_SIZE),
        ("vco", "tuning"): torch.tensor([12.0] * BATCH_SIZE),
        ("vco", "initial_phase"): torch.tensor([0.0] * BATCH_SIZE),
        ("vco", "mod_depth"): torch.tensor([0.0] * BATCH_SIZE),
        ("fm", "tuning"): torch.tensor([0.0] * BATCH_SIZE),
        ("fm", "initial_phase"): torch.tensor([0.0] * BATCH_SIZE),
        ("fm", "mod_depth"): torch.tensor([0.0] * BATCH_SIZE),
    },
    freeze=True,
)
synth = synth.to(device)

synth.randomize(seed=45)
target_audio, _, _ = synth()
ipd.Audio(target_audio[0].cpu().numpy(), rate=SR)

synth.get_parameters()

# Loss
mrstft = auraloss.freq.MultiResolutionSTFTLoss()

# +
# Initialize a TorchSynthModule for optimizing
synth_optim = TorchSynthModule(synth).to(device)

audio = synth_optim()

error = mrstft(target_audio[:, None, :], audio[:, None, :])
print(error)
ipd.Audio(audio[0].detach().cpu().numpy(), rate=SR)
# -

# Parameters Before Optimization
synth.get_parameters()

# +
optimizer = torch.optim.Adam(synth_optim.parameters(), lr=0.005)

pbar = tqdm(range(1000), desc="Iter 0")
for i in pbar:
    optimizer.zero_grad()

    audio = synth_optim()

    error = mrstft(target_audio[:, None, :], audio[:, None, :])
    pbar.set_description(f"Iter {i}: Error: {error}")

    if error < 0.05:
        break
    # error = pitch_loss(target_audio[:,None,:], audio[:,None,:])

    error.backward()
    optimizer.step()
# -

# Parameters After Optimization
synth.get_parameters()

# After optimization
audio = synth_optim()
ipd.Audio(audio[0].detach().cpu().numpy(), rate=SR)
