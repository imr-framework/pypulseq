import math
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
from scipy.optimize import minimize

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
from pypulseq.utils.cumsum import cumsum


def make_extended_trapezoid_area(
    area: float,
    channel: str,
    grad_start: float,
    grad_end: float,
    system: Union[Opts, None] = None,
) -> Tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    """Make (shortest possible) extended trapezoid for given area and gradient start and end point.

    Parameters
    ----------
    area : float
        Area of extended trapezoid.
    channel : str
        Orientation of extended trapezoidal gradient event. Must be one of 'x', 'y' or 'z'.
    grad_start : float
        Starting non-zero gradient value.
    grad_end : float
        Ending non-zero gradient value.
    system: Opts, optional
        System limits.

    Returns
    -------
    grad : SimpleNamespace
        Extended trapezoid event.
    times : numpy.ndarray
        Time points of the extended trapezoid.
    amplitude : numpy.ndarray
        Amplitude values of the extended trapezoid.
    """
    if not system:
        system = Opts()

    slew_rate = system.max_slew * 0.99
    max_grad = system.max_grad * 0.99
    raster_time = system.grad_raster_time

    def _to_raster(time: float) -> float:
        return np.ceil(time / raster_time) * raster_time

    def _calc_area_and_times(grad_amp, flat_time, slew_rate, grad_start, grad_end):
        time_ramp_up = _calc_ramp_time(grad_amp, grad_start, slew_rate)
        time_ramp_down = _calc_ramp_time(grad_amp, grad_end, slew_rate)
        area = _calc_area(grad_amp, time_ramp_up, flat_time, time_ramp_down, grad_start, grad_end)
        return area

    def _calc_area(grad_amp, time_ramp_up, flat_time, time_ramp_down, grad_start, grad_end):
        return (
            0.5 * time_ramp_up * (grad_amp + grad_start)
            + grad_amp * flat_time
            + 0.5 * (grad_amp + grad_end) * time_ramp_down
        )

    def _calc_ramp_time(grad_1, grad_2, slew_rate):
        return abs(grad_1 - grad_2) / slew_rate

    # first try to find grad_amp without a flat time.
    # for better convergence, we ignore the raster effect here and use a very small time step
    # we will take care of raster effect later on
    flat_time = 0

    def obj1(x):
        return (area - _calc_area_and_times(x, flat_time, slew_rate, grad_start, grad_end)) ** 2

    # try different starting points for the optimization
    optimization_results = [
        minimize(fun=obj1, x0=-max_grad, method="Nelder-Mead"),
        minimize(fun=obj1, x0=0, method="Nelder-Mead"),
        minimize(fun=obj1, x0=max_grad, method="Nelder-Mead"),
    ]
    # choose the best result
    optimization_results = np.array([(*res.x, res.fun) for res in optimization_results])
    grad_amp_results, obj1val_results = (
        optimization_results[:, 0],
        optimization_results[:, 1],
    )
    best_index = np.argmin(obj1val_results)
    grad_amp = grad_amp_results[best_index]
    obj1_value = obj1val_results[best_index]

    if obj1_value**0.5 > 1e-3 or abs(grad_amp) > max_grad:
        # Optimization without flat_time did not converge. Try to find flat_time.
        grad_amp = math.copysign(max_grad, grad_amp)

        def obj2(x):
            return (area - _calc_area_and_times(grad_amp, x, slew_rate, grad_start, grad_end)) ** 2

        res2 = minimize(fun=obj2, x0=system.grad_raster_time, method="Nelder-Mead")
        flat_time = res2.x[0]
        # assert flat_time is larger than 0 and that the objective function is small
        assert flat_time >= 0, "flat time is smaller than 0, this should not happen"

    # ensure times are on raster
    flat_time = _to_raster(flat_time)
    time_ramp_up = _to_raster(_calc_ramp_time(grad_amp, grad_start, slew_rate))
    time_ramp_down = _to_raster(_calc_ramp_time(grad_amp, grad_end, slew_rate))

    # recalculate grad_amp while taking raster effect into account
    def obj3(x):
        return (area - _calc_area(x, time_ramp_up, flat_time, time_ramp_down, grad_start, grad_end)) ** 2

    res = minimize(fun=obj3, x0=grad_amp, method="Nelder-Mead")
    grad_amp = res.x[0]

    if flat_time > 0:
        if time_ramp_down > 0:
            if time_ramp_up > 0:
                times = cumsum(0, time_ramp_up, flat_time, time_ramp_down)
                amplitudes = [grad_start, grad_amp, grad_amp, grad_end]
            else:
                times = cumsum(0, flat_time, time_ramp_down)
                amplitudes = [grad_amp, grad_amp, grad_end]
        else:
            times = cumsum(0, time_ramp_up, flat_time)
            amplitudes = [grad_start, grad_amp, grad_end]
    else:
        if time_ramp_down > 0:
            if time_ramp_up > 0:
                times = cumsum(0, time_ramp_up, time_ramp_down)
                amplitudes = np.array([grad_start, grad_amp, grad_end])
            else:
                times = cumsum(0, time_ramp_down)
                amplitudes = np.array([grad_start, grad_end])
        else:
            times = cumsum(0, time_ramp_up)
            amplitudes = np.array([grad_start, grad_end])

    grad = make_extended_trapezoid(channel=channel, system=system, times=times, amplitudes=amplitudes)

    assert abs(grad.area - area) < 1e-3, "Area of the gradient is not equal to the desired area. Optimization failed."

    return grad, np.array(times), amplitudes
