import math
import warnings
from types import SimpleNamespace
from typing import Literal, Union

from pypulseq import eps
from pypulseq.opts import Opts
from pypulseq.utils.tracing import trace, trace_enabled


def calculate_shortest_params_for_area(area: float, max_slew: float, max_grad: float, grad_raster_time: float):
    """Calculate the shortest possible rise_time, flat_time, and fall_time for a given area."""

    # Calculate initial rise time constrained by max slew rate
    rise_time = math.ceil(math.sqrt(abs(area) / max_slew) / grad_raster_time) * grad_raster_time
    rise_time = max(rise_time, grad_raster_time)

    # Calculate initial amplitude
    amplitude = area / rise_time
    effective_time = rise_time

    # Adjust for max gradient constraint
    if abs(amplitude) > max_grad:
        effective_time = math.ceil(abs(area) / max_grad / grad_raster_time) * grad_raster_time
        amplitude = area / effective_time
        rise_time = math.ceil(abs(amplitude) / max_slew / grad_raster_time) * grad_raster_time
        rise_time = max(rise_time, grad_raster_time)

    # Calculate flat and fall times
    flat_time = effective_time - rise_time
    fall_time = rise_time

    return amplitude, rise_time, flat_time, fall_time


def calculate_shortest_rise_time(amplitude: float, max_slew: float, grad_raster_time: float):
    """Calculate the shortest possible rise / fall time for a given amplitude and slew rate."""

    return math.ceil(max(abs(amplitude) / max_slew, grad_raster_time) / grad_raster_time) * grad_raster_time


def make_trapezoid(
    channel: str,
    amplitude: Union[float, None] = None,
    area: Union[float, None] = None,
    delay: float = 0,
    duration: Union[float, None] = None,
    fall_time: Union[float, None] = None,
    flat_area: Union[float, None] = None,
    flat_time: Union[float, None] = None,
    max_grad: Union[float, None] = None,
    max_slew: Union[float, None] = None,
    rise_time: Union[float, None] = None,
    system: Union[Opts, None] = None,
) -> SimpleNamespace:
    """
    Create a trapezoidal gradient event.

    The user must supply any of the following sets of parameters:
    Area based:
    - area
    - area and duration
    - area and duration and rise_time
    - flat_time, area and rise_time
    Amplitude based:
    - amplitude and duration
    - amplitude and flat_time
    Flat area based:
    - flat_area and flat_time
    Additional options may be supplied with the above.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.opts.Opts`

    Parameters
    ----------
    channel : str
        Orientation of trapezoidal gradient event. Must be one of `x`, `y` or `z`.
    amplitude : float, default=None
        Peak amplitude (Hz/m).
    area : float, default=None
        Area (1/m).
    delay : float, default=0
        Delay in seconds (s).
    duration : float, default=None
        Duration in seconds (s). Duration is defined as rise_time + flat_time + fall_time.
    fall_time : float, default=None
        Fall time in seconds (s).
    flat_area : float, default=None
        Flat area (1/m).
    flat_time : float, default=None
        Flat duration in seconds (s). Default is -1 to allow for triangular pulses.
    max_grad : float, default=None
        Maximum gradient strength (Hz/m).
    max_slew : float, default=None
        Maximum slew rate (Hz/m/s).
    rise_time : float, default=0
        Rise time in seconds (s).
    system : Opts, default=Opts()
        System limits.

    Returns
    -------
    grad : SimpleNamespace
        Trapezoidal gradient event created based on the supplied parameters.

    Raises
    ------
    ValueError
        If none of `area`, `flat_area` and `amplitude` are passed
        If requested area is too large for this gradient
        If `flat_time`, `duration` and `area` are not supplied.
        Amplitude violation
    NotImplementedError
        If an input set that might be valid but not implemented is passed.
    """
    if system is None:
        system = Opts.default

    if channel not in ['x', 'y', 'z']:
        raise ValueError(f'Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}')

    if max_grad is None:
        max_grad = system.max_grad

    if max_slew is None:
        max_slew = system.max_slew

    # If either of rise_time or fall_time is not provided, set it to the other.
    rise_time = rise_time or fall_time
    fall_time = fall_time or rise_time

    # Check which one of area, flat_area and amplitude is provided, and determine the calculation path accordingly.
    calc_path: Literal['area', 'flat_area', 'amplitude']
    if area is not None and flat_area is None and amplitude is None:
        calc_path = 'area'
    elif area is None and flat_area is not None and amplitude is None:
        calc_path = 'flat_area'
    elif area is None and flat_area is None and amplitude is not None:
        calc_path = 'amplitude'
    elif area is None and flat_area is not None and amplitude is not None:
        raise NotImplementedError('Flat Area + Amplitude input pair is not implemented yet.')
    elif area is not None and flat_area is None and amplitude is not None:
        raise NotImplementedError('Amplitude + Area input pair is not implemented yet.')
    else:
        raise ValueError("Must supply either 'area', 'flat_area' or 'amplitude'.")

    # Check if sufficient timing parameters are provided.
    if flat_time is not None and flat_area is None and amplitude is None and (rise_time is None or area is None):
        raise ValueError(
            'When `flat_time` is provided, either `flat_area`, '
            'or `amplitude`, or `rise_time` and `area` must be provided as well.'
        )

    if calc_path == 'area':
        # We have area and duration.
        if duration is not None and flat_time is None:
            if rise_time is None:
                _, rise_time, flat_time, fall_time = calculate_shortest_params_for_area(
                    area, max_slew, max_grad, system.grad_raster_time
                )
                min_duration = rise_time + flat_time + fall_time
                assert duration >= min_duration, (
                    f'Requested area is too large for this gradient. Minimum required duration is '
                    f'{round(min_duration * 1e6)} us'
                )

                dc = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
                amplitude2 = (duration - math.sqrt(duration**2 - 4 * abs(area) * dc)) / (2 * dc)
            else:
                if duration <= (rise_time + eps):
                    raise ValueError('The `duration` is too short for the given `rise_time`.')

                if fall_time is None:
                    fall_time = rise_time

                amplitude2 = area / (duration - 0.5 * rise_time - 0.5 * fall_time)
                possible = duration >= (rise_time + fall_time) and abs(amplitude2) <= max_grad
                assert possible, (
                    f'Requested area is too large for this gradient. Probably amplitude is violated '
                    f'{round(abs(amplitude2) / max_grad * 100)}'
                )
            flat_time = duration - rise_time - fall_time
            amplitude2 = area / (rise_time / 2 + fall_time / 2 + flat_time)

        elif flat_time is not None:
            if rise_time is None:
                raise ValueError('Must supply `rise_time` when `area` and `flat_time` is provided.')

            amplitude2 = area / (rise_time + flat_time)

        else:
            if rise_time is not None or fall_time is not None:
                warnings.warn('Rise time and fall time is ignored when calculating the shortest duration from `area`.')

            amplitude2, rise_time, flat_time, fall_time = calculate_shortest_params_for_area(
                area, max_slew, max_grad, system.grad_raster_time
            )

    elif calc_path == 'flat_area':
        # Check and raise invalid input sets
        if duration is not None:
            raise NotImplementedError('Flat Area + Duration input pair is not implemented yet.')
        elif flat_time is not None:
            amplitude2 = flat_area / flat_time

    elif calc_path == 'amplitude':
        if rise_time is None:
            rise_time = abs(amplitude) / max_slew
            rise_time = math.ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
            if rise_time == 0:
                rise_time = system.grad_raster_time
            fall_time = rise_time

        amplitude2 = amplitude
        if duration is not None and flat_time is None:
            flat_time = duration - rise_time - fall_time
        elif flat_time is not None and duration is None:
            pass
        else:
            raise ValueError('Must supply area or duration.')

    if rise_time is None and fall_time is None:
        rise_time = fall_time = calculate_shortest_rise_time(amplitude2, max_slew, system.grad_raster_time)

    if abs(amplitude2) > max_grad:
        raise ValueError(f'Refined amplitude ({abs(amplitude2):0.0f} Hz/m) is larger than max ({max_grad:0.0f} Hz/m).')

    if abs(amplitude2) / rise_time > max_slew:
        raise ValueError(
            f'Refined slew rate ({abs(amplitude2) / rise_time:0.0f} Hz/m/s) for ramp up is larger than max ({max_slew:0.0f} Hz/m/s).'
        )

    if abs(amplitude2) / fall_time > max_slew:
        raise ValueError(
            f'Refined slew rate ({abs(amplitude2) / fall_time:0.0f} Hz/m/s) for ramp down is larger than max ({max_slew:0.0f} Hz/m/s).'
        )

    grad = SimpleNamespace()
    grad.type = 'trap'
    grad.channel = channel
    grad.amplitude = amplitude2
    grad.rise_time = rise_time
    grad.flat_time = flat_time
    grad.fall_time = fall_time
    grad.area = amplitude2 * (flat_time + rise_time / 2 + fall_time / 2)
    grad.flat_area = amplitude2 * flat_time
    grad.delay = delay
    grad.first = 0
    grad.last = 0

    if trace_enabled():
        grad.trace = trace()

    return grad
