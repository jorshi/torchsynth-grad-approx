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

import numpy as np
from scipy.signal.windows import hann
import torch
import torchaudio.functional as fn
from torchsynth.synth import Voice
from torchsynth.config import SynthConfig
import IPython.display as ipd
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa

BATCH_SIZE = 512
DURATION = 1.0

# STFT function
window_a = torch.tensor(hann(1024)).to("cuda")
def stft_a(audio):
    x_stft = torch.stft(audio, n_fft=1024, window=window_a, return_complex=True)
    x_mag = torch.sqrt(
        torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=1e-8)
    )
    return torch.log10(x_mag)


# STFT function 2
window_b = torch.tensor(hann(512)).to("cuda")
def stft_b(audio):
    x_stft = torch.stft(audio, n_fft=512, window=window_b, return_complex=True)
    x_mag = torch.sqrt(
        torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=1e-8)
    )
    return torch.log10(x_mag)


# Load target
sample_rate = 44100
target, _ = librosa.load("./audio/snaredrum.wav", sr=sample_rate)
#target = np.pad(target, [0, int(sample_rate * DURATION - len(target))])
ipd.Audio(target, rate=44100)

# Configure Synth
synth_config = SynthConfig(buffer_size_seconds=len(target)/sample_rate, batch_size=BATCH_SIZE, reproducible=False) 
voice = Voice(synth_config)
# Run on the GPU if it's available
if torch.cuda.is_available():
    voice = voice.to("cuda")

# Spectral loss A
target_spectrogram_a = stft_a(torch.tensor(target).to("cuda"))
def stft_loss_a(audio):
    spectrogram = stft_a(audio)
    return torch.mean(torch.square(spectrogram - target_spectrogram_a), dim=(1,2))


# Spectral loss B
target_spectrogram_b = stft_b(torch.tensor(target).to("cuda"))
def stft_loss_b(audio):
    spectrogram = stft_b(audio)
    return torch.mean(torch.square(spectrogram - target_spectrogram_b), dim=(1,2))


best_audio = None
min_error = np.inf
for i in tqdm(range(25)):
    audio, _, _ = voice(i)
    spectral_error = stft_loss_b(audio)
    min_sample = torch.argmin(spectral_error)
    sample_error = spectral_error[min_sample]
    if sample_error < min_error:
        print(f"New best error: {sample_error}")
        min_error = sample_error
        best_audio = audio[min_sample].cpu().numpy()
    

ipd.Audio(best_audio, rate=sample_rate)

# ## MOGA

import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

num_params = len(list(voice.parameters()))


def fitness(individuals):
    # Update voice with new parameters
    parameters = [param for _, param in sorted(voice.named_parameters())]
    new_values = torch.stack([torch.tensor(list(ind)) for ind in individuals], dim=1).to("cuda")
    for i, parameter in enumerate(parameters):
        parameter.data = new_values[i]
    
    audio, _, _ = voice()
    spectral_error_a = stft_loss_a(audio).cpu().numpy()
    spectral_error_b = stft_loss_b(audio).cpu().numpy()
    
    error = []
    for i in range(len(audio)):
        error.append([spectral_error_a[i], spectral_error_b[i]])
    
    return error


# +
num_objectives = 2
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * num_objectives)
creator.create("Individual", list, fitness=creator.FitnessMin)

# Setup toolbox
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.random)

# Structure initializers
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_float,
    num_params,
)

toolbox.register(
    "population", tools.initRepeat, list, toolbox.individual
)

ref_points = tools.uniform_reference_points(num_objectives, 12)

toolbox.register("evaluate", fitness)

toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=30.0)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=0.0,
    up=1.0,
    eta=20.0,
    indpb=1.0 / num_params,
)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

# +
# Create the initial population
pop = toolbox.population(n=BATCH_SIZE)
#invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.evaluate(pop)
for i, ind in enumerate(pop):
    ind.fitness.values = fitnesses[i]
    
logbook = tools.Logbook()

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

logbook.header = "gen", "evals", "std", "min", "avg", "max"

record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
# -

pbar = tqdm(range(1, 1000 + 1), desc="Generation 1")
for gen in pbar:
    offspring = algorithms.varAnd(pop, toolbox, 0.5, 0.5)
    
    fitnesses = toolbox.evaluate(offspring)
    for i, ind in enumerate(offspring):
        ind.fitness.values = fitnesses[i]
        
    # Select the next generation population from parents and offspring
    pop = toolbox.select(pop + offspring, BATCH_SIZE)
    
    # Compile statistics about the new population
    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(offspring), **record)
    pbar.set_description(f"Generation {gen}: Min {record['min']}")

# +
tools.selBest(pop, BATCH_SIZE)
parameters = [param for _, param in sorted(voice.named_parameters())]
new_values = torch.stack([torch.tensor(list(ind)) for ind in tools.selBest(pop, BATCH_SIZE)], dim=1).to("cuda")
for i, parameter in enumerate(parameters):
    parameter.data = new_values[i]

audio, _, _ = voice()
# -

ipd.Audio(audio[510].cpu().numpy(), rate=sample_rate)


