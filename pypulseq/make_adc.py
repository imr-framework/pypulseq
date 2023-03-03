from types import SimpleNamespace

from pypulseq.opts import Opts


def make_adc(
    num_samples: int,
    delay: float = 0,
    duration: float = 0,
    dwell: float = 0,
    freq_offset: float = 0,
    phase_offset: float = 0,
    system: Opts = Opts(),
) -> SimpleNamespace:
    """
    Create an ADC readout event.

    Parameters
    ----------
    num_samples: int
        Number of readout samples.
    system : Opts, default=Opts()
        System limits. Default is a system limits object initialised to default values.
    dwell : float, default=0
        ADC dead time in seconds (s) after sampling.
    duration : float, default=0
        Duration in seconds (s) of ADC readout event with `num_samples` number of samples.
    delay : float, default=0
        Delay in seconds (s) of ADC readout event.
    freq_offset : float, default=0
        Frequency offset of ADC readout event.
    phase_offset : float, default=0
        Phase offset of ADC readout event.

    Returns
    -------
    adc : SimpleNamespace
        ADC readout event.

    Raises
    ------
    ValueError
        If neither `dwell` nor `duration` are defined.
    """
    adc = SimpleNamespace()
    adc.type = "adc"
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

    if adc.dead_time > adc.delay:
        adc.delay = adc.dead_time

    return adc
