import warnings
from types import SimpleNamespace
from typing import Union

import numpy as np

from pypulseq import eps
from pypulseq.opts import Opts
from pypulseq.utils.tracing import trace, trace_enabled


def make_arbitrary_grad(
    channel: str,
    waveform: np.ndarray,
    first: Union[float, None] = None,
    last: Union[float, None] = None,
    delay: float = 0,
    max_grad: Union[float, None] = None,
    max_slew: Union[float, None] = None,
    system: Union[Opts, None] = None,
    oversampling: bool = False,
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
    first : float, default=None
        Gradient value at the start of the gradient event. (t=0)
        Will default to a linear extrapolated value if not provided.
    last : float, default=None
        Gradient value at the end of the gradient event. (t=duration)
        Will default to a linear extrapolated value if not provided.
    delay : float, default=0
        Delay in seconds (s).
    max_grad : float, default=None
        Maximum gradient strength.
        Will default to `system.max_grad` if not provided.
    max_slew : float, default=None
        Maximum slew rate.
        Will default to `system.max_slew` if not provided.
    system : Opts, default=Opts()
        System limits.
        Will default to `pypulseq.opts.default` if not provided.
    oversampling : bool, default=False
        Boolean flag to indicate if gradient is oversampled by a factor of 2.

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

    if channel not in ['x', 'y', 'z']:
        raise ValueError(f'Invalid channel. Must be one of x, y or z. Passed: {channel}')

    if first is None:
        warnings.warn(
            'it will be compulsory to provide the first point of the gradient shape in the future releases; finding the first by extrapolation for now...',
            FutureWarning,
        )
        if oversampling:
            first = 2 * waveform[0] - waveform[1]  # extrapolate by 1 gradient raster
        else:
            first = (3 * waveform[0] - waveform[1]) * 0.5  # extrapolate by 1/2 gradient of the raster

    if last is None:
        warnings.warn(
            'it will be compulsory to provide the last point of the gradient shape in the future releases; finding the last by extrapolation for now...',
            FutureWarning,
        )
        if oversampling:
            last = 2 * waveform[-1] - waveform[-2]  # extrapolate by 1 gradient raster
        else:
            last = (3 * waveform[-1] - waveform[-2]) * 0.5  # extrapolate by 1/2 gradient of the raster

    # Slew rate calculation
    if oversampling:
        slew_rate = np.concatenate([[first - waveform[0]], np.diff(waveform), [last - waveform[-1]]]) / (
            system.grad_raster_time * 2
        )
    else:
        slew_rate = (
            np.concatenate([[2 * (first - waveform[0])], np.diff(waveform), [2 * (waveform[-1] - last)]])
            / system.grad_raster_tim
        )
    if max(abs(slew_rate)) > max_slew * (1 + eps):
        raise ValueError(f'Slew rate violation {max(abs(slew_rate)) / max_slew * 100}')
    if max(abs(waveform)) > max_grad + eps:
        raise ValueError(f'Gradient amplitude violation {max(abs(waveform)) / max_grad * 100}')

    grad = SimpleNamespace()
    grad.type = 'grad'
    grad.channel = channel
    grad.waveform = waveform
    grad.delay = delay
    if oversampling:
        if len(waveform) % 2 == 0:
            raise ValueError('When oversampling is active, waveform must have an odd number of samples')
        grad.area = (waveform[::2] * system.grad_raster_time).sum()
        grad.tt = np.arange(1, len(waveform) + 1) * 0.5 * system.grad_raster_time
        grad.shape_dur = (len(waveform) + 1) * 0.5 * system.grad_raster_time
    else:
        grad.area = (waveform * system.grad_raster_time).sum()
        grad.tt = (np.arange(len(waveform)) + 0.5) * system.grad_raster_time
        grad.shape_dur = len(waveform) * system.grad_raster_time
    grad.first = first
    grad.last = last

    if trace_enabled():
        grad.trace = trace()

    return grad
