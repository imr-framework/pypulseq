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
    system: Union[Opts, None] = None,
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

    Raises
    ------
        ValueError if no solution was found that satisfies the constraints and the desired area.
    """
    if not system:
        system = Opts()

    max_slew = system.max_slew * 0.99
    max_grad = system.max_grad * 0.99
    raster_time = system.grad_raster_time

    def _to_raster(time: float) -> float:
        return np.ceil(time / raster_time) * raster_time

    def _calc_ramp_time(grad_1: float, grad_2: float) -> float:
        return _to_raster(abs(grad_1 - grad_2) / max_slew)

    def _find_solution(duration: int) -> Union[None, Tuple[int, int, int, float]]:
        """Find extended trapezoid gradient waveform for given duration.

        The function performs a grid search over all possible ramp-up, ramp-down and flat times
        for the given duration and returns the solution with the lowest slew rate.

        Parameters
        ----------
        duration
            duration of the gradient in integer multiples of raster_time

        Returns
        -------
            Tuple of ramp-up time, flat time, ramp-down time, gradient amplitude or None if no solution was found
        """

        # Calculate possible ramp-up times for given duration
        ramp_time_pos_start = round(_calc_ramp_time(max_grad, grad_start) / raster_time)
        ramp_time_neg_start = round(_calc_ramp_time(-max_grad, grad_start) / raster_time)
        max_ramp_time_start = max(ramp_time_pos_start, ramp_time_neg_start)
        min_time_start = min(max_ramp_time_start, duration - 1)
        time_ramp_up = np.arange(1, min_time_start + 1)

        # Calculate possible ramp-down times for given duration
        ramp_time_pos_end = round(_calc_ramp_time(max_grad, grad_end) / raster_time)
        ramp_time_neg_end = round(_calc_ramp_time(-max_grad, grad_end) / raster_time)
        max_ramp_time_end = max(ramp_time_pos_end, ramp_time_neg_end)
        min_time_end = min(max_ramp_time_end, duration - 1)
        time_ramp_down = np.arange(1, min_time_end + 1)

        # Create meshgrid of possible ramp-up and ramp-down times
        time_ramp_up, time_ramp_down = np.meshgrid(time_ramp_up, time_ramp_down)

        # calculate corresponding flat times
        flat_time = duration - time_ramp_up - time_ramp_down

        # filter search space for valid timings (flat time >= 0)
        valid_indices = flat_time >= 0
        time_ramp_up = time_ramp_up[valid_indices]
        time_ramp_down = time_ramp_down[valid_indices]
        flat_time = flat_time[valid_indices]

        # Calculate gradient strength for given timing using analytical solution
        grad_amp = -(time_ramp_up * raster_time * grad_start + time_ramp_down * raster_time * grad_end - 2 * area) / (
            (time_ramp_up + 2 * flat_time + time_ramp_down) * raster_time
        )

        # Calculate slew rates for given timings
        slew_rate1 = abs(grad_start - grad_amp) / (time_ramp_up * raster_time)
        slew_rate2 = abs(grad_end - grad_amp) / (time_ramp_down * raster_time)

        # Filter solutions that satisfy max_grad and max_slew constraints
        valid_indices = (
            (abs(grad_amp) <= max_grad + 1e-8) & (slew_rate1 <= max_slew + 1e-8) & (slew_rate2 <= max_slew + 1e-8)
        )
        solutions = np.flatnonzero(valid_indices)

        # Check if any valid solutions were found
        if solutions.shape[0] == 0:
            return None

        # Find solution with lowest slew rate and return it
        ind = np.argmin(slew_rate1[valid_indices] + slew_rate2[valid_indices])
        ind = solutions[ind]
        return (int(time_ramp_up[ind]), int(flat_time[ind]), int(time_ramp_down[ind]), float(grad_amp[ind]))

    # Perform a linear search
    # This is necessary because there can exist a dead space where solutions
    # do not exist for some durations longer than the optimal duration. The
    # binary search below fails to find the optimum in those cases.
    # TODO: Check if range is sufficient, try to calculate the dead space.
    min_duration = max(round(_calc_ramp_time(grad_end, grad_start) / raster_time), 2)
    max_duration = (
        max(
            round(_calc_ramp_time(max_grad, grad_start) / raster_time),
            round(_calc_ramp_time(-max_grad, grad_start) / raster_time),
            round(_calc_ramp_time(max_grad, grad_end) / raster_time),
            round(_calc_ramp_time(-max_grad, grad_end) / raster_time),
            min_duration,
        )
        + 1
    )

    solution = None
    for duration in range(min_duration, max_duration):
        solution = _find_solution(duration)
        if solution:
            break

    # Perform a binary search if no solution was found
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

            test_value = (upper_limit + lower_limit) // 2

            if fun(test_value):
                return binary_search(fun, lower_limit, test_value)
            else:
                return binary_search(fun, test_value, upper_limit)

        solution = binary_search(_find_solution, max_duration // 2, max_duration)

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

    if not abs(grad.area - area) < 1e-8:
        raise ValueError(f"Could not find a solution for area={area}.")

    return grad, np.array(times), amplitudes
