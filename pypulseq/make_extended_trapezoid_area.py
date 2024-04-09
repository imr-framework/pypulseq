from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
from pypulseq.utils.cumsum import cumsum


def make_extended_trapezoid_area(
    area: float,
    channel: str,
    grad_start: float,
    grad_end: float,
    system: Union[Opts, None] = None
) -> Tuple[SimpleNamespace, np.array, np.array]:
    """Make the shortest possible extended trapezoid for given area and gradient start and end point.

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
        
    max_slew = system.max_slew * 0.99
    max_grad = system.max_grad * 0.99
    raster_time = system.grad_raster_time

    def _to_raster(time: float) -> float:
        return np.ceil(time / raster_time) * raster_time

    def _calc_ramp_time(grad_1, grad_2):
        return _to_raster(abs(grad_1 - grad_2) / max_slew)
    
    # Find a solution for a given duration (specified in integer amounts of raster_time)
    def _find_solution(duration):
        # Grid search over all possible ramp-up and ramp-down times
        # TODO: It should be possible to prune this search space significantly
        #       based on the reachable gradient amplitudes and the target area.
        time_ramp_up = np.arange(1, min(
                                        max(round(_calc_ramp_time(max_grad, grad_start) / raster_time),
                                            round(_calc_ramp_time(-max_grad, grad_start) / raster_time))
                                        , duration-1)
                                        +1)
        time_ramp_down = np.arange(1, min(
                                        max(round(_calc_ramp_time(max_grad, grad_end) / raster_time),
                                            round(_calc_ramp_time(-max_grad, grad_end) / raster_time))
                                        , duration-1)
                                    +1)
        
        time_ramp_up, time_ramp_down = np.meshgrid(time_ramp_up, time_ramp_down)
        
        # Calculate flat time and filter search space for valid timing
        flat_time = duration - time_ramp_up - time_ramp_down
        mask = flat_time >= 0
        
        time_ramp_up = time_ramp_up[mask]
        time_ramp_down = time_ramp_down[mask]
        flat_time = flat_time[mask]
        
        # Calculate gradient strength for given timing
        grad_amp = -(time_ramp_up*raster_time*grad_start + time_ramp_down*raster_time*grad_end - 2*area)/((time_ramp_up + 2*flat_time + time_ramp_down)*raster_time)
        
        # Filter solutions that satisfy max_grad and max_slew
        slew_rate1 = abs(grad_start - grad_amp) / (time_ramp_up * raster_time)
        slew_rate2 = abs(grad_end - grad_amp) / (time_ramp_down * raster_time)
        mask = (abs(grad_amp) <= max_grad + 1e-8) & (slew_rate1 <= max_slew + 1e-8) & (slew_rate2 <= max_slew + 1e-8)
        
        solutions = np.flatnonzero(mask)
        
        # Check if there are solutions
        if solutions.shape[0] == 0:
            return None
        
        # Find solution with lowest slew rate and return it
        ind = np.argmin(slew_rate1[mask] + slew_rate2[mask])
        ind = solutions[ind]
        return (time_ramp_up[ind], flat_time[ind], time_ramp_down[ind], grad_amp[ind])
         
    
    # Perform a linear search
    # This is necessary because there can exist a dead space where solutions
    # do not exist for some durations longer than the optimal duration. The
    # binary search below fails to find the optimum in those cases.
    # TODO: Check that this range is sufficient, and maybe find out if the dead
    #       space can be calculated.
    min_duration = max(round(_calc_ramp_time(grad_end, grad_start) / raster_time), 2)
    max_duration = max(round(_calc_ramp_time(max_grad, grad_start) / raster_time),
                       round(_calc_ramp_time(-max_grad, grad_start) / raster_time),
                       round(_calc_ramp_time(max_grad, grad_end) / raster_time),
                       round(_calc_ramp_time(-max_grad, grad_end) / raster_time),
                       min_duration) + 1

    solution = None
    for duration in range(min_duration, max_duration):
        solution = _find_solution(duration)
        if solution:
            break

    # Perform a binary search
    if not solution:
        max_duration = duration
        
        # First, find the upper limit on duration where a solution exists by
        # exponentially expanding the duration.
        while not solution:
            max_duration *= 2
            solution = _find_solution(max_duration)
    
        def binary_search(fun, lower_limit, upper_limit):           
            if lower_limit == upper_limit - 1:
                return fun(upper_limit)
            
            test_value = (upper_limit + lower_limit)//2
            
            if fun(test_value):
                return binary_search(fun, lower_limit, test_value)
            else:
                return binary_search(fun, test_value, upper_limit)
    
        solution = binary_search(_find_solution, max_duration//2, max_duration)

    # Get timing and gradient amplitude from solution
    time_ramp_up = solution[0] * raster_time
    flat_time = solution[1] * raster_time
    time_ramp_down = solution[2] * raster_time
    grad_amp = solution[3]

    # Create extended trapezoid
    if flat_time > 0:
        times = cumsum(0, time_ramp_up, flat_time, time_ramp_down)
        amplitudes = np.array([grad_start, grad_amp, grad_amp, grad_end])
    else:
        times = cumsum(0, time_ramp_up, time_ramp_down)
        amplitudes = np.array([grad_start, grad_amp, grad_end])

    grad = make_extended_trapezoid(channel=channel, system=system, times=times, amplitudes=amplitudes)

    assert abs(grad.area - area) < 1e-3, "Area of the gradient is not equal to the desired area. Optimization failed."

    return grad, np.array(times), amplitudes
