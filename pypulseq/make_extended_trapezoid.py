from types import SimpleNamespace

import numpy as np

from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform


def make_extended_trapezoid(channel: str, times: np.ndarray = np.zeros(1), amplitudes: np.ndarray = np.zeros(1),
                            system: Opts = Opts(), max_grad: float = 0, max_slew: float = 0, skip_check: bool = False):
    """
    Creates an extend trapezoidal gradient event by defined by amplitude values in `amplitudes` at time indices in
    `times`.

    Parameters
    ----------
    channel : str
        Orientation of extended trapezoidal gradient event. Must be one of x, y or z.
    times : numpy.ndarray, optional
        Time points at which `amplitudes` defines amplitude values. Default is 0.
    amplitudes : numpy.ndarray, optional
        Values defined at `times` time indices. Default is 0.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    max_grad : float, optional
        Maximum gradient strength. Default is 0.
    max_slew : float, optional
        Maximum slew rate. Default is 0.
    skip_check : bool, optional
        Perform check. Default is false.

    Returns
    -------
    grad : SimpleNamespace
        Extended trapezoid gradient event.
    """
    if channel not in ['x', 'y', 'z']:
        raise ValueError()

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
