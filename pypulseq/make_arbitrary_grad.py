from types import SimpleNamespace

import numpy as np

from pypulseq.opts import Opts


def make_arbitrary_grad(channel: str, waveform: np.ndarray, system: Opts = Opts(), max_grad: float = 0,
                        max_slew: float = 0, delay: float = 0) -> SimpleNamespace:
    """
    Creates a gradient event with arbitrary waveform.

    Parameters
    ----------
    channel : str
        Orientation of gradient event of arbitrary shape. Must be one of `x`, `y` or `z`.
    waveform : numpy.ndarray
        Arbitrary waveform.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
     max_grad : float, optional
        Maximum gradient strength. Default is 0.
     max_slew : float, optional
        Maximum slew rate. Default is 0.
    delay : float, optional
        Delay in milliseconds (ms).

    Returns
    -------
    grad : SimpleNamespace
        Gradient event with arbitrary waveform.
    """
    if channel not in ['x', 'y', 'z']:
        raise ValueError(f'Invalid channel. Must be one of x, y or z. You passed: {channel}')

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    g = waveform
    slew = np.squeeze(np.subtract(g[1:], g[:-1]) / system.grad_raster_time)
    if max(abs(slew)) >= max_slew:
        raise ValueError(f'Slew rate violation {max(abs(slew)) / max_slew * 100}')
    if max(abs(g)) >= max_grad:
        raise ValueError(f'Gradient amplitude violation {max(abs(g)) / max_grad * 100}')

    grad = SimpleNamespace()
    grad.type = 'grad'
    grad.channel = channel
    grad.waveform = g
    grad.delay = delay
    grad.t = np.arange(len(g)) * system.grad_raster_time
    grad.first = g[0]
    grad.last = g[-1]

    return grad
