from types import SimpleNamespace
from typing import Union

import numpy as np

from pypulseq.opts import Opts


def make_arbitrary_grad(
    channel: str,
    waveform: np.ndarray,
    delay: float = 0,
    max_grad: Union[float, None] = None,
    max_slew: Union[float, None] = None,
    system: Union[Opts, None] = None,
) -> SimpleNamespace:
    """
    Creates a gradient event from an arbitrary waveform.

    Note that the sample points are assumed to be equally spaced by `system.grad_raster_time`
    and that the given waveform values are the values in the middle of each raster interval.

    The duration of the gradient is thus given by the number of samples times `system.grad_raster_time`.

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
    channel : str
        Orientation of gradient event of arbitrary shape. Must be one of `x`, `y` or `z`.
    waveform : numpy.ndarray
        Arbitrary waveform.
    system : Opts
        System limits.
        Will default to `pypulseq.opts.default` if not provided.
    max_grad : float
        Maximum gradient strength.
        Will default to `system.max_grad` if not provided.
    max_slew : float
        Maximum slew rate.
        Will default to `system.max_slew` if not provided.
    delay : float, default=0
        Delay in seconds (s).

    Returns
    -------
    grad : SimpleNamespace
        Gradient event with arbitrary waveform.

    Raises
    ------
    ValueError
        If invalid `channel` is passed. Must be one of x, y or z.
        If slew rate is violated.
        If gradient amplitude is violated.
    """
    if system is None:
        system = Opts.default

    if max_grad is None or max_grad == 0:
        max_grad = system.max_grad

    if max_slew is None or max_slew == 0:
        max_slew = system.max_slew

    if channel not in ["x", "y", "z"]:
        raise ValueError(f"Invalid channel. Must be one of x, y or z. Passed: {channel}")

    slew_rate = np.diff(waveform) / system.grad_raster_time
    if max(abs(slew_rate)) >= max_slew:
        raise ValueError(f"Slew rate violation {max(abs(slew_rate)) / max_slew * 100}")
    if max(abs(waveform)) >= max_grad:
        raise ValueError(f"Gradient amplitude violation {max(abs(waveform)) / max_grad * 100}")

    grad = SimpleNamespace()
    grad.type = "grad"
    grad.channel = channel
    grad.waveform = waveform
    grad.delay = delay
    grad.tt = (np.arange(len(waveform)) + 0.5) * system.grad_raster_time
    grad.shape_dur = len(waveform) * system.grad_raster_time
    grad.first = (3 * waveform[0] - waveform[1]) * 0.5  # Extrapolate by 1/2 gradient raster
    grad.last = (waveform[-1] * 3 - waveform[-2]) * 0.5  # Extrapolate by 1/2 gradient raster
    grad.area = (waveform * system.grad_raster_time).sum()

    return grad
