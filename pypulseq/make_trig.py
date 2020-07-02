# inserted for trigger support by mveldmann

from types import SimpleNamespace
from pypulseq.opts import Opts

def make_trig(delay: float = 0, duration: float = 1e-5 , system: Opts = Opts()) -> SimpleNamespace:
    """
    Creates a trigger event.

    Parameters
    ----------
    delay : float
        Delay in seconds
    duration: float
        Duration in seconds

    Returns
    -------
    trigger : SimpleNamespace
        Trigger event.
    """

    trig = SimpleNamespace()
    if delay < 0:
        raise ValueError('Delay {:.2f} ms is invalid'.format(delay * 1e3))
    trig.type = 'trigger'
    trig.delay = delay
    if duration < system.grad_raster_time:
        print('Duration too short, set to gradient raster time (minimum duration)')
        trig.duration = system.grad_raster_time
    else:
        trig.duration = duration
    return trig
