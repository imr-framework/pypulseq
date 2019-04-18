from decimal import *

from pypulseq.holder import Holder
from pypulseq.opts import Opts


def makeadc(kwargs):
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

    num_samples = kwargs.get("num_samples", 0)
    system = kwargs.get("system", Opts())
    dwell = kwargs.get("dwell", 0)
    duration = kwargs.get("duration", 0)
    delay = kwargs.get("delay", 0)
    freq_offset = kwargs.get("freq_offset", 0)
    phase_offset = kwargs.get("phase_offset", 0)

    adc = Holder()
    adc.type = 'adc'
    adc.num_samples = num_samples
    adc.dwell = dwell
    adc.delay = delay
    adc.freq_offset = freq_offset
    adc.phase_offset = phase_offset
    adc.dead_time = system.adc_dead_time

    if (dwell == 0 and duration == 0) or (dwell > 0 and duration > 0):
        raise ValueError("Either dwell or duration must be defined, not both")

    if duration > 0:
        # getcontext().prec = 4
        adc.dwell = float(Decimal(duration) / Decimal(num_samples))
    adc.duration = dwell * num_samples if dwell > 0 else 0
    return adc
