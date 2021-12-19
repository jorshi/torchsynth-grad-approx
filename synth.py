"""
Torchsynth Synths for experiments
"""

from typing import Optional

import torch
from torchsynth.module import (
    ADSR,
    ControlRateUpsample,
    FmVCO,
    MonophonicKeyboard,
    SineVCO,
    SquareSawVCO,
    VCA,
)
from torchsynth.synth import AbstractSynth
from torchsynth.config import SynthConfig


class SimpleSynth(AbstractSynth):
    """
    A Simple Synthesizer with a SquareSaw oscillator
    and an ADSR modulating the amplitude

    Args:
        synthconfig: Synthesizer configuration that defines the
            batch_size, buffer_size, and sample_rate among other
            variables that control synthesizer functioning
    """

    def __init__(self, synthconfig: Optional[SynthConfig] = None):

        # Make sure to call __init__ in the parent AbstractSynth
        super().__init__(synthconfig=synthconfig)

        # These are all the modules that we are going to use.
        # Pass in a list of tuples with (name, SynthModule,
        # optional params dict) after we add them we will be
        # able to access them as attributes with the same name.
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard),
                ("adsr", ADSR),
                ("upsample", ControlRateUpsample),
                ("vco", SquareSawVCO),
                ("vca", VCA),
            ]
        )

    def output(self) -> torch.Tensor:
        """
        This is called when we trigger the synth. We link up
        all the individual modules and pass the outputs through
        to the output of this method.
        """
        # Keyboard is parameter module, it returns parameter
        # values for the midi_f0 note value and the duration
        # that note is held for.
        midi_f0, note_on_duration = self.keyboard()

        # The amplitude envelope is generated based on note duration
        envelope = self.adsr(note_on_duration)

        # The envelope that we get from ADSR is at the control rate,
        # which is by default 100x less than the sample rate. This
        # reduced control rate is used for performance reasons.
        # We need to upsample the envelope prior to use with the VCO output.
        envelope = self.upsample(envelope)

        # Generate SquareSaw output at frequency for the midi note
        out = self.vco(midi_f0)

        # Apply the amplitude envelope to the oscillator output
        out = self.vca(out, envelope)

        return out


class SimpleFMSynth(AbstractSynth):
    """
    A Simple FM Synthesizer

    Args:
        synthconfig: Synthesizer configuration that defines the
            batch_size, buffer_size, and sample_rate among other
            variables that control synthesizer functioning
    """

    def __init__(self, synthconfig: Optional[SynthConfig] = None):

        # Make sure to call __init__ in the parent AbstractSynth
        super().__init__(synthconfig=synthconfig)

        # These are all the modules that we are going to use.
        # Pass in a list of tuples with (name, SynthModule,
        # optional params dict) after we add them we will be
        # able to access them as attributes with the same name.
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard),
                ("adsr", ADSR),
                ("upsample", ControlRateUpsample),
                ("vco", SineVCO),
                ("fm", FmVCO),
                ("vca", VCA),
            ]
        )

    def output(self) -> torch.Tensor:
        """
        This is called when we trigger the synth. We link up
        all the individual modules and pass the outputs through
        to the output of this method.
        """
        # Keyboard is parameter module, it returns parameter
        # values for the midi_f0 note value and the duration
        # that note is held for.
        midi_f0, note_on_duration = self.keyboard()

        # The amplitude envelope is generated based on note duration
        envelope = self.adsr(note_on_duration)

        # The envelope that we get from ADSR is at the control rate,
        # which is by default 100x less than the sample rate. This
        # reduced control rate is used for performance reasons.
        # We need to upsample the envelope prior to use with the VCO output.
        envelope = self.upsample(envelope)

        # Generate SquareSaw output at frequency for the midi note
        out = self.vco(midi_f0)
        out = self.fm(midi_f0, out)

        # Apply the amplitude envelope to the oscillator output
        out = self.vca(out, envelope)

        return out
