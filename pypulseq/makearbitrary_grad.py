import numpy as np

from pypulseq.holder import Holder
from pypulseq.opts import Opts


def makearbitrary_grad(kwargs):
    """
    Makes a Holder object for an arbitrary gradient Event.

    Parameters
    ----------
    kwargs : dict
        Key value mappings of RF Event parameters_params and values.

    Returns
    -------
    grad : Holder
        Trapezoidal gradient Event configured based on supplied kwargs.
    """

    channel = kwargs.get("channel", "z")
    system = kwargs.get("system", Opts())
    waveform = kwargs.get("waveform")
    max_grad = kwargs.get("max_grad") if kwargs.get("max_grad", 0) > 0 else system.max_grad
    max_slew = kwargs.get("max_slew ") if kwargs.get("max_slew ", 0) > 0 else system.max_slew

    g = np.reshape(waveform, (1, -1))
    slew = (g[0][1:] - g[0][0:-1]) / system.grad_raster_time
    if max(abs(slew)) > max_slew:
        raise ValueError('Slew rate violation {:f}'.format(max(abs(slew)) / max_slew * 100))
    if max(abs(g[0])) > max_grad:
        raise ValueError('Gradient amplitude violation {:f}'.format(max(abs(g)) / max_grad * 100))
    grad = Holder()
    grad.type = 'grad'
    grad.channel = channel
    grad.waveform = g
    grad.t = np.arange(len(g[0])) * system.grad_raster_time
    grad.t = np.reshape(grad.t, (1, -1))
    return grad
