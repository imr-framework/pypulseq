from decimal import *

from types import SimpleNamespace
from pypulseq.opts import Opts


def make_adc(num_samples=0, system=Opts(), dwell=0, duration=0, delay=0, freq_offset=0, phase_offset=0):
    """
    Makes a Holder object for an ADC Event.

    Parameters
    ----------
    kwargs : dict
        Key value mappings of ADC Event parameters_params and values.

    Returns
    -------
    adc : Holder
        ADC Event.
    """
    adc = SimpleNamespace()
    adc.type = 'adc'
    adc.num_samples = num_samples
    adc.dwell = dwell
    adc.delay = delay
    adc.freq_offset = freq_offset
    adc.phase_offset = phase_offset
    adc.dead_time = system.adc_dead_time

    if (dwell == 0 and duration == 0) or (dwell > 0 and duration > 0):
        raise ValueError("Either dwell or duration must be defined")

    if duration > 0:
        adc.dwell = duration / num_samples

    if dwell > 0:
        adc.duration = dwell * num_samples

    return adc
