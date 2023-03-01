import math
from types import SimpleNamespace
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts


def make_extended_trapezoid_area(
    area: float, channel: str, grad_end: float, grad_start: float, system: Opts
) -> Tuple[SimpleNamespace, np.array, np.array]:
    """
    Makes the shortest possible extended trapezoid with a given area which starts and ends (optionally) as non-zero
    gradient values.

    Parameters
    ----------
    channel : str
        Orientation of extended trapezoidal gradient event. Must be one of 'x', 'y' or 'z'.
    grad_start : float
        Starting non-zero gradient value.
    grad_end : float
        Ending non-zero gradient value.
    area : float
        Area of extended trapezoid.
    system: Opts
        System limits.

    Returns
    -------
    grad : SimpleNamespace
        Extended trapezoid event.
    times : numpy.ndarray
    amplitude : numpy.ndarray
    """
    SR = system.max_slew * 0.99

    Tp = 0
    obj1 = (
        lambda x: (
            area - __testGA(x, 0, SR, system.grad_raster_time, grad_start, grad_end)
        )
        ** 2
    )
    arr_res = [
        minimize(fun=obj1, x0=-system.max_grad, method="Nelder-Mead"),
        minimize(fun=obj1, x0=0, method="Nelder-Mead"),
        minimize(fun=obj1, x0=system.max_grad, method="Nelder-Mead"),
    ]
    arr_res = np.array([(*res.x, res.fun) for res in arr_res])
    Gp, obj1val = arr_res[:, 0], arr_res[:, 1]
    i_min = np.argmin(obj1val)
    Gp = Gp[i_min]
    obj1val = obj1val[i_min]

    if obj1val > 1e-3 or np.abs(Gp) > system.max_grad:  # Search did not converge
        Gp = system.max_grad * np.sign(Gp)
        obj2 = (
            lambda x: (
                area
                - __testGA(Gp, x, SR, system.grad_raster_time, grad_start, grad_end)
            )
            ** 2
        )
        res2 = minimize(fun=obj2, x0=0, method="Nelder-Mead")
        T, obj2val = *res2.x, res2.fun
        assert obj2val < 1e-2

        Tp = np.ceil(T / system.grad_raster_time) * system.grad_raster_time

        # Fix the ramps
        Tru = (
            np.ceil(np.abs(Gp - grad_start) / SR / system.grad_raster_time)
            * system.grad_raster_time
        )
        Trd = (
            np.ceil(np.abs(Gp - grad_end) / SR / system.grad_raster_time)
            * system.grad_raster_time
        )
        obj3 = lambda x: (area - __testGA1(x, Tru, Tp, Trd, grad_start, grad_end)) ** 2

        res = minimize(fun=obj3, x0=Gp, method="Nelder-Mead")
        Gp, obj3val = *res.x, res.fun
        assert obj3val < 1e-3  # Did the final search converge?

    assert Tp >= 0

    if Tp > 0:
        times = np.cumsum([0, Tru, Tp, Trd])
        amplitudes = [grad_start, Gp, Gp, grad_end]
    else:
        Tru = (
            np.ceil(np.abs(Gp - grad_start) / SR / system.grad_raster_time)
            * system.grad_raster_time
        )
        Trd = (
            np.ceil(np.abs(Gp - grad_end) / SR / system.grad_raster_time)
            * system.grad_raster_time
        )

        if Trd > 0:
            if Tru > 0:
                times = np.cumsum([0, Tru, Trd])
                amplitudes = np.array([grad_start, Gp, grad_end])
            else:
                times = np.cumsum([0, Trd])
                amplitudes = np.array([grad_start, grad_end])
        else:
            times = np.cumsum([0, Tru])
            amplitudes = np.array([grad_start, grad_end])

    grad = make_extended_trapezoid(
        channel=channel, system=system, times=times, amplitudes=amplitudes
    )
    grad.area = __testGA1(Gp, Tru, Tp, Trd, grad_start, grad_end)

    assert np.abs(grad.area - area) < 1e-3

    return grad, times, amplitudes


def __testGA(Gp, Tp, SR, dT, Gs, Ge):
    Tru = np.ceil(np.abs(Gp - Gs) / SR / dT) * dT
    Trd = np.ceil(np.abs(Gp - Ge) / SR / dT) * dT
    ga = __testGA1(Gp, Tru, Tp, Trd, Gs, Ge)
    return ga


def __testGA1(Gp, Tru, Tp, Trd, Gs, Ge):
    return 0.5 * Tru * (Gp + Gs) + Gp * Tp + 0.5 * (Gp + Ge) * Trd
