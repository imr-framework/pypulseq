import math
from types import SimpleNamespace
from typing import Union

import numpy as np

from pypulseq.opts import Opts
from pypulseq.utils.tracing import trace, trace_enabled


def calculate_shortest_params_for_area(area, max_slew, max_grad, grad_raster_time):
    rise_time = math.ceil(math.sqrt(abs(area) / max_slew) / grad_raster_time) * grad_raster_time
    if rise_time < grad_raster_time:  # Area was almost 0 maybe
        rise_time = grad_raster_time
    amplitude = np.divide(area, rise_time)  # To handle nan
    t_eff = rise_time

    if abs(amplitude) > max_grad:
        t_eff = math.ceil(abs(area) / max_grad / grad_raster_time) * grad_raster_time
        amplitude = area / t_eff
        rise_time = math.ceil(abs(amplitude) / max_slew / grad_raster_time) * grad_raster_time

        if rise_time == 0:
            rise_time = grad_raster_time

    flat_time = t_eff - rise_time
    fall_time = rise_time

    return amplitude, rise_time, flat_time, fall_time


def make_trapezoid(
    channel: str,
    amplitude: float = 0,
    area: Union[float, None] = None,
    delay: float = 0,
    duration: float = 0,
    fall_time: float = 0,
    flat_area: Union[float, None] = None,
    flat_time: float = -1,
    max_grad: float = 0,
    max_slew: float = 0,
    rise_time: float = 0,
    system: Union[Opts, None] = None,
) -> SimpleNamespace:
    """
    Create a trapezoidal gradient event.

    The user must supply as a minimum one of the following sets:
    - area
    - amplitude and duration
    - flat_time and flat_area
    - flat_time and amplitude
    - flat_time, area and rise_time
    Additional options may be supplied with the above.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.opts.Opts`

    Parameters
    ----------
    channel : str
        Orientation of trapezoidal gradient event. Must be one of `x`, `y` or `z`.
    amplitude : float, default=0
        Peak amplitude (Hz/m).
    area : float, default=None
        Area (1/m).
    delay : float, default=0
        Delay in seconds (s).
    duration : float, default=0
        Duration in seconds (s). Duration is defined as rise_time + flat_time + fall_time.
    fall_time : float, default=0
        Fall time in seconds (s).
    flat_area : float, default=0
        Flat area (1/m).
    flat_time : float, default=-1
        Flat duration in seconds (s). Default is -1 to allow for triangular pulses.
    max_grad : float, default=0
        Maximum gradient strength (Hz/m).
    max_slew : float, default=0
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
    """
    if system is None:
        system = Opts.default

    if channel not in ['x', 'y', 'z']:
        raise ValueError(f'Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}')

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    if rise_time <= 0:
        rise_time = 0.0

    if fall_time > 0:
        if rise_time == 0:
            raise ValueError(
                'Invalid arguments. Must always supply `rise_time` if `fall_time` is specified explicitly.'
            )
    else:
        fall_time = 0.0

    if area is None and flat_area is None and amplitude == 0:
        raise ValueError("Must supply either 'area', 'flat_area' or 'amplitude'.")

    if flat_time != -1:
        if amplitude != 0:
            amplitude2 = amplitude
        elif area is not None and rise_time > 0:
            # We have rise_time, flat_time and area.
            amplitude2 = area / (rise_time + flat_time)
        elif flat_area is not None:
            amplitude2 = flat_area / flat_time
        else:
            raise ValueError(
                'When `flat_time` is provided, either `flat_area`, '
                'or `amplitude`, or `rise_time` and `area` must be provided as well.'
            )

        if rise_time == 0:
            rise_time = abs(amplitude2) / max_slew
            rise_time = math.ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
            if rise_time == 0:
                rise_time = system.grad_raster_time
        if fall_time == 0:
            fall_time = rise_time
    elif duration > 0:
        if amplitude == 0:
            if rise_time == 0:
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
                if fall_time == 0:
                    fall_time = rise_time
                amplitude2 = area / (duration - 0.5 * rise_time - 0.5 * fall_time)
                possible = duration >= (rise_time + fall_time) and abs(amplitude2) <= max_grad
                assert possible, (
                    f'Requested area is too large for this gradient. Probably amplitude is violated '
                    f'{round(abs(amplitude) / max_grad * 100)}'
                )
        else:
            amplitude2 = amplitude

        if rise_time == 0:
            rise_time = math.ceil(abs(amplitude2) / max_slew / system.grad_raster_time) * system.grad_raster_time
            if rise_time == 0:
                rise_time = system.grad_raster_time

        if fall_time == 0:
            fall_time = rise_time
        flat_time = duration - rise_time - fall_time

        if amplitude == 0:
            # Adjust amplitude (after rounding) to match area
            amplitude2 = area / (rise_time / 2 + fall_time / 2 + flat_time)
    else:
        if area is None:
            raise ValueError('Must supply area or duration.')
        else:
            # Find the shortest possible duration.
            amplitude2, rise_time, flat_time, fall_time = calculate_shortest_params_for_area(
                area, max_slew, max_grad, system.grad_raster_time
            )

    assert (
        abs(amplitude2) <= max_grad
    ), f'Refined amplitude ({abs(amplitude2):0.0f} Hz/m) is larger than max ({max_grad:0.0f} Hz/m).'

    assert (
        abs(amplitude2) / rise_time <= max_slew
    ), f'Refined slew rate ({abs(amplitude2)/rise_time:0.0f} Hz/m/s) for ramp up is larger than max ({max_slew:0.0f} Hz/m/s).'

    assert (
        abs(amplitude2) / fall_time <= max_slew
    ), f'Refined slew rate ({abs(amplitude2)/fall_time:0.0f} Hz/m/s) for ramp down is larger than max ({max_slew:0.0f} Hz/m/s).'

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
