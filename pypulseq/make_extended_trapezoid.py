from types import SimpleNamespace
from typing import Iterable

import numpy as np

from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform


def make_extended_trapezoid(channel: str, amplitudes: Iterable = np.zeros(1), max_grad: float = 0,
                            max_slew: float = 0, system: Opts = Opts(), skip_check: bool = False,
                            times: Iterable = np.zeros(1)) -> SimpleNamespace:
    """
    Creates an extend trapezoidal gradient event by defined by amplitude values in `amplitudes` at time indices in
    `times`.

    Parameters
    ----------
    channel : str
        Orientation of extended trapezoidal gradient event. Must be one of 'x', 'y' or 'z'.
    amplitudes : numpy.ndarray, optional, default=09
        Values defined at `times` time indices.
    max_grad : float, optional, default=0
        Maximum gradient strength.
    max_slew : float, optional, default=0
        Maximum slew rate.
    system : Opts, optional, default=Opts()
        System limits.
    skip_check : bool, optional, default=False
        Perform check.
    times : numpy.ndarray, optional, default=np.zeros(1)
        Time points at which `amplitudes` defines amplitude values.

    Returns
    -------
    grad : SimpleNamespace
        Extended trapezoid gradient event.

    Raises
    ------
    ValueError
        If invalid `channel` is passed. Must be one of 'x', 'y' or 'z'.
        If all elements in `times` are zero.
        If elements in `times` are not in ascending order or not distinct.
        If all elements in `amplitudes` are zero.
        If first amplitude of a gradient is non-ero and does not connect to a previous block.
    """
    if channel not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid channel. Must be one of 'x', 'y' or 'z'. Passed: {channel}")

    if not np.any(times):
        raise ValueError('At least one of the given times must be non-zero')

    if np.any(np.diff(times) <= 0):
        raise ValueError('Times must be in ascending order and all times must be distinct')

    if not np.any(amplitudes):
        raise ValueError('At least one of the given amplitudes must be non-zero')

    if skip_check is False and times[0] > 0 and amplitudes[0] != 0:
        raise ValueError('If first amplitude of a gradient is non-zero, it must connect to previous block')

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    waveform = points_to_waveform(times=times, amplitudes=amplitudes, grad_raster_time=system.grad_raster_time)
    grad = make_arbitrary_grad(channel=channel, waveform=waveform, system=system, max_grad=max_grad, max_slew=max_slew,
                               delay=times[0])
    grad.first = amplitudes[0]
    grad.last = amplitudes[-1]

    return grad
