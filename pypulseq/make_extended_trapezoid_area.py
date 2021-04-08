import math
from types import SimpleNamespace
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts


def make_extended_trapezoid_area(channel: str, Gs: float, Ge: float, A: float,
                                 system: Opts) -> Tuple[SimpleNamespace, np.array, np.array]:
    """
    Makes shortest possible extended trapezoid with a given area.

    Parameters
    ----------
    channel : str
        Orientation of extended trapezoidal gradient event. Must be one of 'x', 'y' or 'z'.
    Gs : float
        Starting non-zero gradient value.
    Ge : float
        Ending non-zero gradient value.
    A : float
        Area of extended trapezoid.
    system: Opts
        System limits.

    Returns
    -------
    grad : SimpleNamespace
        Extended trapezoid event.
    times : numpy.ndarray
    amplitude : numpy.ndarray

    Raises
    ------
    ValueError

    """
    SR = system.max_slew * 0.99

    Tp = 0
    obj1 = lambda x: (A - __testGA(x, 0, SR, system.grad_raster_time, Gs, Ge)) ** 2
    res = minimize(fun=obj1, x0=0, method='Nelder-Mead')
    Gp, obj1val = *res.x, res.fun

    if obj1val > 1e-3 or abs(Gp) > system.max_grad:  # Search did not converge
        Gp = system.max_grad * np.sign(Gp)
        obj2 = lambda x: (A - __testGA(Gp, x, SR, system.grad_raster_time, Gs, Ge)) ** 2
        res2 = minimize(fun=obj2, x0=0, method='Nelder-Mead')
        T, obj2val = *res2.x, res2.fun
        assert obj2val < 1e-2

        Tp = math.ceil(T / system.grad_raster_time) * system.grad_raster_time

        # Fix the ramps
        Tru = math.ceil(abs(Gp - Gs) / SR / system.grad_raster_time) * system.grad_raster_time
        Trd = math.ceil(abs(Gp - Ge) / SR / system.grad_raster_time) * system.grad_raster_time
        obj3 = lambda x: (A - __testGA1(x, Tru, Tp, Trd, Gs, Ge)) ** 2

        res = minimize(fun=obj3, x0=Gp, method='Nelder-Mead')
        Gp, obj3val = *res.x, res.fun
        assert obj3val < 1e-3

    if Tp > 0:
        times = np.cumsum([0, Tru, Tp, Trd])
        amplitudes = [Gs, Gp, Gp, Ge]
    else:
        Tru = math.ceil(abs(Gp - Gs) / SR / system.grad_raster_time) * system.grad_raster_time
        Trd = math.ceil(abs(Gp - Ge) / SR / system.grad_raster_time) * system.grad_raster_time

        if Trd > 0:
            if Tru > 0:
                times = np.cumsum([0, Tru, Trd])
                amplitudes = np.array([Gs, Gp, Ge])
            else:
                times = np.cumsum([0, Trd])
                amplitudes = np.array([Gs, Ge])
        else:
            times = np.cumsum([0, Tru])
            amplitudes = np.array([Gs, Ge])

    grad = make_extended_trapezoid(channel=channel, system=system, times=times, amplitudes=amplitudes)

    return grad, times, amplitudes


def __testGA(Gp, Tp, SR, dT, Gs, Ge):
    Tru = math.ceil(abs(Gp - Gs) / SR / dT) * dT
    Trd = math.ceil(abs(Gp - Ge) / SR / dT) * dT
    ga = __testGA1(Gp, Tru, Tp, Trd, Gs, Ge)
    return ga


def __testGA1(Gp, Tru, Tp, Trd, Gs, Ge):
    return 0.5 * Tru * (Gp + Gs) + Gp * Tp + 0.5 * (Gp + Ge) * Trd
