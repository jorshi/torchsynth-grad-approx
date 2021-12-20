# Optimizing synthesizer parameters using gradient approximation
### NASH 2021 Hackathon!

[![Open in
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorshi/torchsynth-grad-approx/blob/main/optimize_simple_synth.ipynb)

These are some experiments I conducted during [NASH 2021](https://signas-qmul.github.io/nash/),
the Neural Audio Synthesis Hackathon that took place on the 18th & 19th of December.

Over the weekend I explored implementing gradient approximation for 
[torchsynth](https://github.com/torchsynth/torchsynth), so that synthesizers could
be included in deep learning models & training without having to have the full synth
be differentiable. It uses simultaneous perturbation stochastic approximation
(SPSA) to estimate the gradients for synthesizer parameters. This technique was
used by Marco A. Martínez Ramírez et al. in their work on [Differentiable Signal
Processing With Black-Box Audio Effects](https://arxiv.org/abs/2105.04752).

I was able to start optimizing on a few parameters for a simple synthesizer, but ran
into issues as soon as oscillator tuning or FM was introduced. There is a known issue
with audio loss functions for calculating loss with pitch 
([Turian and Henry, 2020](https://arxiv.org/abs/2012.04572)), so this is not surprising.

Nonetheless, techniques like SPSA seem promising for including traditional DSP synthesis
into neural nets and deep learning!

Fun weekend puttering around with this! Thank you to Ben Hayes for organing the event.
