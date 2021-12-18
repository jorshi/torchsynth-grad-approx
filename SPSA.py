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
from torchsynth.config import SynthConfig
from torchsynth.synth import AbstractSynth, Voice
import IPython.display as ipd

BATCH_SIZE = 128
SR = 44100

# Configure Synth
synth_config = SynthConfig(batch_size=BATCH_SIZE, sample_rate=SR, reproducible=False) 
voice = Voice(synth_config)
# Run on the GPU if it's available
if torch.cuda.is_available():
    voice = voice.to("cuda")


# +
class TorchSynth(nn.Module):
    
    def __init__(self, synth: AbstractSynth):
        super(TorchSynth, self).__init__()
        self.synth = synth
        self.parameters = [param for _, param in sorted(self.synth.named_parameters())]
        
    
    def forward(self, input):
        """
        Input is a set of parameters for the synth
        """
        for i, parameter in enumerate(self.parameters):
            parameter.data = input[:,i]

        return self.synth()
        

    
# -

synth = TorchSynth(voice)

audio, params, _ = synth(torch.rand(128, 78).to("cuda"))

ipd.Audio(audio[0].cpu().numpy(), rate=SR)


