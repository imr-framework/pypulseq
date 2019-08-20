from types import SimpleNamespace

from pypulseq.opts import Opts


def make_adc(num_samples: int = 0, system: Opts = Opts(), dwell: float = 0, duration: float = 0, delay: float = 0,
             freq_offset: float = 0, phase_offset: float = 0) -> SimpleNamespace:
    """
    Creates an ADC readout event.

    Parameters
    ----------
    num_samples: int, optional
        Number of readout samples.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    dwell : float, optional
        ADC dead time in milliseconds (ms) after sampling.
    duration : float, optional
        Duration in milliseconds (ms) of ADC readout event with `num_samples` number of samples.
    delay : float, optional
        Delay in milliseconds (ms) of ADC readout event.
    freq_offset : float, optional
        Frequency offset of ADC readout event.
    phase_offset : float, optional
        Phase offset of ADC readout event.

    Returns
    -------
    adc : SimpleNamespace
        ADC readout event.
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
