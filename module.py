"""
SPSA Synth Modules
"""

import torch
import torch.nn as nn
from torchsynth.synth import AbstractSynth


class TorchSynthSPSA(torch.autograd.Function):
    @staticmethod
    def play_synth(input, synth):
        parameters = [param for _, param in synth.get_parameters().items()]
        for i, parameter in enumerate(parameters):
            parameter.requires_grad = False
            parameter.data = torch.clip(input[:, i], 0.0, 1.0)

        output, _, _ = synth()

        return output

    @staticmethod
    def forward(ctx, input, synth):
        ctx.save_for_backward(input)
        ctx.synth = synth
        return TorchSynthSPSA.play_synth(input, synth)

    @staticmethod
    def perturbation(y):
        """
        Random samples from a Rademacher distribution - same shape as the input tensor
        """
        x = torch.empty_like(y).uniform_(0, 1)
        x = torch.bernoulli(x)
        # Keep x where x==1, otherwise change to -1.0
        x = torch.where(x == 1.0, x, torch.tensor(-1.0, dtype=y.dtype, device=y.device))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Initialize the gradient on the input to None
        input = ctx.saved_tensors[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            # Need to compute the gradient w.r.t input (synth parameters)

            # Initialize gradient for input
            grad_input = torch.empty_like(input)

            # Simultaneous Pertubation Stochastic Approximation
            eps = 1e-4
            delta = TorchSynthSPSA.perturbation(input)

            j_plus = TorchSynthSPSA.play_synth(input + eps * delta, ctx.synth)
            j_minus = TorchSynthSPSA.play_synth(input - eps * delta, ctx.synth)

            grady_num = j_plus - j_minus

            # Iterate through each parameter and compute gradient
            for i in range(input.shape[1]):
                # Dot product between the output grad for each batch
                for j in range(input.shape[0]):
                    grady = grady_num / (2 * eps * delta[j, i])
                    grad_input[j, i] = torch.dot(grad_output[j], grady[j])

        return grad_input, None


class TorchSynthModule(nn.Module):
    """
    Module that contains randomly initialized synth parameters
    """

    def __init__(self, synth: AbstractSynth):
        super(TorchSynthModule, self).__init__()
        self.synth = synth
        self.num_parameters = len(synth.get_parameters())
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
