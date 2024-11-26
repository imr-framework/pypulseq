from types import SimpleNamespace
from typing import Union

from pypulseq.opts import Opts


def make_digital_output_pulse(
    channel: str, delay: float = 0, duration: float = 4e-3, system: Union[Opts, None] = None
) -> SimpleNamespace:
    """
    Create a digital output pulse event a.k.a. trigger. Creates an output trigger event on a given channel with optional
    given delay and duration.

    Parameters
    ----------
    channel : str
        Must be one of 'osc0','osc1', or 'ext1'.
    delay : float, default=0
        Delay in seconds (s).
    duration : float, default=4e-3
        Duration of trigger event in seconds (s).
    system : Opts, default=Opts()
        System limits.

    Returns
    -------
    trig : SimpleNamespace
        Trigger event.

    Raises
    ------
    ValueError
        If `channel` is invalid. Must be one of 'osc0','osc1', or 'ext1'.
    """
    if system is None:
        system = Opts.default

    if channel not in ['osc0', 'osc1', 'ext1']:
        raise ValueError(f"Channel {channel} is invalid. Must be one of 'osc0','osc1', or 'ext1'.")

    trig = SimpleNamespace()
    trig.type = 'output'
    trig.channel = channel
    trig.delay = delay
    trig.duration = duration
    if trig.duration <= system.grad_raster_time:
        trig.duration = system.grad_raster_time

    return trig
