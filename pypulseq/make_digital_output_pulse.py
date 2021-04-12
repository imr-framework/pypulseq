from types import SimpleNamespace

from pypulseq.opts import Opts


def make_digital_output_pulse(channel: str, delay: float = 0, duration: float = 0,
                              system: Opts = Opts()) -> SimpleNamespace:
    """
    Create a digital output pulse event a.k.a. trigger. Creates an output trigger event on a given channel with optional
    given delay and duration.

    Parameters
    ----------
    channel : str
        Must be one of 'osc0','osc1', or 'ext1'.
    delay : float, optional, default=0
        Delay, in millis.
    duration : float, optional, default=0
        Duration of trigger event, in millis.
    system : Opts, optional, default=Opts()
        System limits.

    Returns
    ------
    trig : SimpleNamespace
        Trigger event.

    Raises
    ------
    ValueError
        If `channel` is invalid. Must be one of 'osc0','osc1', or 'ext1'.
    """
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
