from copy import deepcopy
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from pypulseq import eps
from pypulseq.calc_duration import calc_duration
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform


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
    # copy() to emulate pass-by-value; otherwise passed grad events are modified
    grads = deepcopy(grads)

    if max_grad <= 0:
        max_grad = system.max_grad
    if max_slew <= 0:
        max_slew = system.max_slew

    if len(grads) < 2:
        raise ValueError("Cannot add less than two gradients")

    # First gradient defines channel
    channel = grads[0].channel

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
            tt_rast = grads[ii].tt / system.grad_raster_time + 0.5
            is_arb.append(np.all(np.abs(tt_rast - np.arange(len(tt_rast)))) < eps)

    # Convert to numpy.ndarray for fancy-indexing later on
    firsts, lasts = np.array(firsts), np.array(lasts)

    common_delay = np.min(delays)
    total_duration = np.max(durs)

    # Check if we have a set of traps with the same timing
    if np.all(is_trap):
        cond1 = 1 == len(np.unique([g.delay for g in grads]))
        cond2 = 1 == len(np.unique([g.rise_time for g in grads]))
        cond3 = 1 == len(np.unique([g.flat_time for g in grads]))
        cond4 = 1 == len(np.unique([g.fall_time for g in grads]))
        if cond1 and cond2 and cond3 and cond4:
            grad = grads[0]
            grad.amplitude = np.sum([g.amplitude for g in grads])
            grad.area = np.sum([g.area for g in grads])
            grad.flat_area = np.sum([g.flat_area for g in grads])

            return grad

    # Check if we only have arbitrary grads on irregular time samplings, optionally mixed with trapezoids
    if np.all(np.logical_or(is_trap, np.logical_not(is_arb))):
        # Keep shapes still rather simple
        times = []
        for ii in range(len(grads)):
            g = grads[ii]
            if g.type == "trap":
                times.extend(
                    np.cumsum([g.delay, g.rise_time, g.flat_time, g.fall_time])
                )
            else:
                times.extend(g.delay + g.tt)

        times = np.sort(np.unique(times))
        dt = times[1:] - times[:-1]
        ieps = dt < eps
        if np.any(ieps):
            dtx = [times[0], *dt]
            dtx[ieps] = (
                dtx[ieps] + dtx[ieps + 1]
            )  # Assumes that no more than two too similar values can occur
            dtx[ieps + 1] = []
            times = np.cumsum(dtx)

        amplitudes = np.zeros_like(times)
        for ii in range(len(grads)):
            g = grads[ii]
            if g.type == "trap":
                if g.flat_time > 0:  # Trapezoid or triangle
                    g.tt = np.cumsum([0, g.rise_time, g.flat_time, g.fall_time])
                    g.waveform = [0, g.amplitude, g.amplitude, 0]
                else:
                    g.tt = np.cumsum([0, g.rise_time, g.fall_time])
                    g.waveform = [0, g.amplitude, 0]

            tt = g.delay + g.tt
            # Fix rounding for the first and last time points
            i_min = np.argmin(np.abs(tt[0] - times))
            t_min = (np.abs(tt[0] - times))[i_min]
            if t_min < eps:
                tt[0] = times[i_min]
            i_min = np.argmin(np.abs(tt[-1] - times))
            t_min = (np.abs(tt[-1] - times))[i_min]
            if t_min < eps:
                tt[-1] = times[i_min]

            if np.abs(g.waveform[0]) > eps and tt[0] > eps:
                tt[0] += eps

            amplitudes += np.interp(xp=tt, fp=g.waveform, x=times)

        grad = make_extended_trapezoid(
            channel=channel, amplitudes=amplitudes, times=times, system=system
        )
        return grad

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
        max_length = num_points if num_points > max_length else max_length

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
    # Last is defined by the sum of lasts with the maximum duration (total_duration)
    grad.first = np.sum(firsts[np.array(delays) == common_delay])
    grad.last = np.sum(lasts[np.where(durs == total_duration)])

    return grad
