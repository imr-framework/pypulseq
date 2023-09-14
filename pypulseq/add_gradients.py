from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from pypulseq import eps
from pypulseq.calc_duration import calc_duration
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.utils.cumsum import cumsum

def add_gradients(
    grads: Iterable[SimpleNamespace],
    max_grad: int = 0,
    max_slew: int = 0,
    system=Opts(),
) -> SimpleNamespace:
    """
    Returns the superposition of several gradients.

    Parameters
    ----------
    grads : [SimpleNamespace, ...]
        Gradient events.
    system : Opts, default=Opts()
        System limits.
    max_grad : float, default=0
        Maximum gradient amplitude.
    max_slew : float, default=0
        Maximum slew rate.

    Returns
    -------
    grad : SimpleNamespace
        Superimposition of gradient events from `grads`.
    """
    if max_grad <= 0:
        max_grad = system.max_grad
    if max_slew <= 0:
        max_slew = system.max_slew

    if len(grads) == 0:
        raise ValueError("No gradients specified")
    if len(grads) == 1:
        # Trapezoids only require a shallow copy
        if grads[0].type == 'trap':
            return copy(grads[0])
        else:
            return deepcopy(grads[0])
    
    # First gradient defines channel
    channel = grads[0].channel

    # Check if we have a set of traps with the same timing
    if all(g.type == 'trap' for g in grads):
        cond1 = all(g.delay == grads[0].delay for g in grads)
        cond2 = all(g.rise_time == grads[0].rise_time for g in grads)
        cond3 = all(g.flat_time == grads[0].flat_time for g in grads)
        cond4 = all(g.fall_time == grads[0].fall_time for g in grads)
        
        if cond1 and cond2 and cond3 and cond4:
            return make_trapezoid(grads[0].channel,
                                  amplitude=sum(g.amplitude for g in grads)+eps,
                                  rise_time=grads[0].rise_time,
                                  flat_time=grads[0].flat_time,
                                  fall_time=grads[0].fall_time,
                                  delay=grads[0].delay,
                                  system=system)
    
    # Find out the general delay of all gradients and other statistics
    delays, firsts, lasts, durs, is_trap, is_arb = [], [], [], [], [], []
    for ii in range(len(grads)):
        if grads[ii].channel != channel:
            raise ValueError("Cannot add gradients on different channels.")

        delays.append(grads[ii].delay)
        firsts.append(grads[ii].first)
        lasts.append(grads[ii].last)
        durs.append(calc_duration(grads[ii]))
        is_trap.append(grads[ii].type == "trap")
        if is_trap[-1]:
            is_arb.append(False)
        else:
            tt_rast = grads[ii].tt / system.grad_raster_time - 0.5
            is_arb.append(np.all(np.abs(tt_rast - np.arange(len(tt_rast)))) < eps)

    # Check if we only have arbitrary grads on irregular time samplings, optionally mixed with trapezoids
    if np.all(np.logical_or(is_trap, np.logical_not(is_arb))):
        # Keep shapes still rather simple
        times = []
        for ii in range(len(grads)):
            g = grads[ii]
            if g.type == "trap":
                times.extend(
                    cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time)
                )
            else:
                times.extend(g.delay + g.tt)

        times = np.sort(np.unique(times))
        dt = times[1:] - times[:-1]
        ieps = np.flatnonzero(dt < eps)
        if np.any(ieps):
            dtx = np.array([times[0], *dt])
            dtx[ieps] = (
                dtx[ieps] + dtx[ieps + 1]
            )  # Assumes that no more than two too similar values can occur
            dtx = np.delete(dtx, ieps + 1)
            times = np.cumsum(dtx)

        amplitudes = np.zeros_like(times)
        for ii in range(len(grads)):
            g = grads[ii]
            if g.type == "trap":
                if g.flat_time > 0:  # Trapezoid or triangle
                    tt = list(cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time))
                    waveform = [0, g.amplitude, g.amplitude, 0]
                else:
                    tt = list(cumsum(g.delay, g.rise_time, g.fall_time))
                    waveform = [0, g.amplitude, 0]
            else:
                tt = g.delay + g.tt
                waveform = g.waveform

            # Fix rounding for the first and last time points
            i_min = np.argmin(np.abs(tt[0] - times))
            t_min = (np.abs(tt[0] - times))[i_min]
            if t_min < eps:
                tt[0] = times[i_min]
            i_min = np.argmin(np.abs(tt[-1] - times))
            t_min = (np.abs(tt[-1] - times))[i_min]
            if t_min < eps:
                tt[-1] = times[i_min]

            if abs(waveform[0]) > eps and tt[0] > eps:
                tt[0] += eps

            amplitudes += np.interp(xp=tt, fp=waveform, x=times)

        grad = make_extended_trapezoid(
            channel=channel, amplitudes=amplitudes, times=times, system=system
        )
        return grad
    
    # Convert to numpy.ndarray for fancy-indexing later on
    firsts, lasts = np.array(firsts), np.array(lasts)
    common_delay = np.min(delays)
    durs = np.array(durs)

    # Convert everything to a regularly-sampled waveform
    waveforms = dict()
    max_length = 0
    for ii in range(len(grads)):
        g = grads[ii]
        if g.type == "grad":
            if is_arb[ii]:
                waveforms[ii] = g.waveform
            else:
                waveforms[ii] = points_to_waveform(
                    amplitudes=g.waveform,
                    times=g.tt,
                    grad_raster_time=system.grad_raster_time,
                )
        elif g.type == "trap":
            if g.flat_time > 0:  # Triangle or trapezoid
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.flat_time,
                        g.delay
                        - common_delay
                        + g.rise_time
                        + g.flat_time
                        + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, g.amplitude, 0])
            else:
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, 0])
            waveforms[ii] = points_to_waveform(
                amplitudes=amplitudes,
                times=times,
                grad_raster_time=system.grad_raster_time,
            )
        else:
            raise ValueError("Unknown gradient type")

        if g.delay - common_delay > 0:
            # Stop for numpy.arange is not g.delay - common_delay - system.grad_raster_time like in Matlab
            # so as to include the endpoint
            t_delay = np.arange(0, g.delay - common_delay, step=system.grad_raster_time)
            waveforms[ii] = np.insert(waveforms[ii], 0, t_delay)

        num_points = len(waveforms[ii])
        max_length = max(num_points, max_length)

    w = np.zeros(max_length)
    for ii in range(len(grads)):
        wt = np.zeros(max_length)
        wt[0 : len(waveforms[ii])] = waveforms[ii]
        w += wt

    grad = make_arbitrary_grad(
        channel=channel,
        waveform=w,
        system=system,
        max_slew=max_slew,
        max_grad=max_grad,
        delay=common_delay,
    )
    # Fix the first and the last values
    # First is defined by the sum of firsts with the minimal delay (common_delay)
    # Last is defined by the sum of lasts with the maximum duration (total_duration == durs.max())
    grad.first = np.sum(firsts[np.array(delays) == common_delay])
    grad.last = np.sum(lasts[durs == durs.max()])

    return grad
