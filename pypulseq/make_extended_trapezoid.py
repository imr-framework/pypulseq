import numpy as np

from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform


def make_extended_trapezoid(channel, times=0, amplitudes=0, system=Opts(), max_grad=0, max_slew=0, skip_check=False):
    if channel not in ['x', 'y', 'z']:
        raise ValueError()

    if not np.any(times):
        raise ValueError('At least one of the given times must be non-zero')

    if np.any(np.diff(times) <= 0):
        raise ValueError('Times must be in ascending order and all times mut be distinct')

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
