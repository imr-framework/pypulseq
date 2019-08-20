from types import SimpleNamespace

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform


def add_gradients(grads: list, system=Opts(), max_grad: int = 0, max_slew: int = 0) -> np.ndarray:
    """
    Superpose several gradient events.

    Parameters
    ----------
    grads : list
        List of 'SimpleNamespace' gradient events.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    max_grad : float, optional
        Maximum gradient amplitude. Default is 0.
    max_slew : float, optional
        Maximum slew rate. Default is 0.

    Returns
    -------
    grad : SimpleNamespace
        Superimposition of gradient events from `grads`.
    """
    max_grad = max_grad if max_grad > 0 else system.max_grad
    max_slew = max_slew if max_slew > 0 else system.max_slew

    if len(grads) < 2:
        raise Exception()

    # First gradient defines channel
    channel = grads[0].channel

    # Find out the general delay of all gradients and other statistics
    delays, firsts, lasts, durs = [], [], [], []
    for ii in range(len(grads)):
        delays.append(grads[ii].delay)
        firsts.append(grads[ii].first)
        lasts.append(grads[ii].last)
        durs.append(calc_duration(grads[ii]))

    # Convert to numpy.ndarray for fancy-indexing later on
    firsts, lasts = np.array(firsts), np.array(lasts)

    common_delay = min(delays)
    total_duration = max(durs)

    waveforms = dict()
    max_length = 0
    for ii in range(len(grads)):
        g = grads[ii]
        if g.type == 'grad':
            waveforms[ii] = g.waveform
        elif g.type == 'trap':
            if g.flat_time > 0:  # Triangle or trapezoid
                times = [g.delay - common_delay,
                         g.delay - common_delay + g.rise_time,
                         g.delay - common_delay + g.rise_time + g.flat_time,
                         g.delay - common_delay + g.rise_time + g.flat_time + g.fall_time]
                amplitudes = [0, g.amplitude, g.amplitude, 0]
            else:
                times = [g.delay - common_delay,
                         g.delay - common_delay + g.rise_time,
                         g.delay - common_delay + g.rise_time + g.flat_time]
                amplitudes = [0, g.amplitude, 0]
            waveforms[ii] = points_to_waveform(times=times, amplitudes=amplitudes,
                                               grad_raster_time=system.grad_raster_time)
        else:
            raise ValueError('Unknown gradient type')

        if g.delay - common_delay > 0:
            # Stop for numpy.arange is not g.delay - common_delay - system.grad_raster_time like in Matlab
            # so as to include the endpoint
            t_delay = np.arange(0, g.delay - common_delay, step=system.grad_raster_time)
            waveforms[ii] = np.insert(waveforms[ii], 0, t_delay)

        num_points = len(waveforms[ii])
        max_length = num_points if num_points > max_length else max_length

    w = np.zeros(max_length)
    for ii in range(len(grads)):
        wt = np.zeros(max_length)
        wt[0:len(waveforms[ii])] = waveforms[ii]
        w += wt

    grad = make_arbitrary_grad(channel, w, system, max_slew=max_slew, max_grad=max_grad, delay=common_delay)
    grad.first = np.sum(firsts[np.where(delays == common_delay)])
    grad.last = np.sum(lasts[np.where(durs == total_duration)])

    return grad
