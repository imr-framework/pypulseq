from types import SimpleNamespace

import numpy as np

from pypulseq.opts import Opts


def make_arbitrary_grad(channel, waveform, system=Opts(), max_grad=0, max_slew=0, delay=0):
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

    if channel not in ['x', 'y', 'z']:
        raise ValueError()

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    g = waveform
    slew = (g[1:] - g[:-1]) / system.grad_raster_time
    if max(abs(slew)) >= max_slew:
        raise ValueError('Slew rate violation {:f}'.format(max(abs(slew)) / max_slew * 100))
    if max(abs(g)) >= max_grad:
        raise ValueError('Gradient amplitude violation {:f}'.format(max(abs(g)) / max_grad * 100))

    grad = SimpleNamespace()
    grad.type = 'grad'
    grad.channel = channel
    grad.waveform = g
    grad.t = np.arange(len(g)) * system.grad_raster_time
    grad.first = g[0]
    grad.last = g[-1]

    return grad
