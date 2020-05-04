import math
from types import SimpleNamespace

from pypulseq.opts import Opts


def make_trapezoid(channel='z', system=Opts(), duration=0, area=-1, flat_time=None, flat_area=None, amplitude=None,
                   max_grad=0, max_slew=0, rise_time=0, delay=0):
    """
    Creates a trapezoidal gradient event.

    Parameters
    ----------
    channel : str, optional
        Orientation of trapezoidal gradient event. Must be one of `x`, `y` or `z`.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    duration : float, optional
        Duration in milliseconds (ms). Default is 0.
    area : float, optional
        Area. Default is -1.
    flat_time : float, optional
        Flat duration in milliseconds (ms). Default is `None`.
    flat_area : float, optional
        Flat area. Default is `None`.
    amplitude : float, optional
        Amplitude. Default is `None`.
    max_grad : float, optional
        Maximum gradient strength. Default is 0.
    max_slew : float, optional
        Maximum slew rate. Default is 0.
    rise_time : float, optional
        Rise time in milliseconds (ms). Default is 0.
    delay : float, optional
        Delay in milliseconds (ms). Default is 0.

    Returns
    -------
    grad : SimpleNamespace
        Trapezoidal gradient event created based on the supplied parameters.
    """
    if channel not in ['x', 'y', 'z']:
        raise ValueError()

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    if rise_time <= 0:
        rise_time = system.rise_time

    if area is None and flat_area is None and amplitude is None:
        raise ValueError('Must supply either \'area\', \'flat_area\' or \'amplitude\'')

    if flat_time is not None:
        if amplitude is not None:
            amplitude2 = amplitude
        else:
            amplitude2 = flat_area / flat_time

        if rise_time is None:
            rise_time = abs(amplitude2) / max_slew
            rise_time = math.ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
        fall_time, flat_time = rise_time, flat_time
    elif duration > 0:
        amplitude2 = amplitude
        if amplitude is None:
            if rise_time is None:
                dC = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
                possible = duration ** 2 > 4 * abs(area) * dC
                amplitude2 = (duration - math.sqrt(duration ** 2 - 4 * abs(area) * dC)) / (2 * dC)
            else:
                amplitude2 = area / (duration - rise_time)
                possible = duration > 2 * rise_time and abs(amplitude2) < max_grad

            if not possible:
                raise ValueError('Requested area is too large for this gradient')

        if rise_time is None:
            rise_time = math.ceil(
                abs(amplitude2) / max_slew / system.grad_raster_time) * system.grad_raster_time

        fall_time = rise_time
        flat_time = duration - rise_time - fall_time

        if amplitude is None:
            amplitude2 = area / (rise_time / 2 + fall_time / 2 + flat_time)
    else:
        if area is None:
            raise ValueError('Must supply a duration')
        else:
            rise_time = math.ceil(math.sqrt(abs(area) / max_slew) / system.grad_raster_time) * system.grad_raster_time
            amplitude2 = area / rise_time
            t_eff = rise_time

            if abs(amplitude2) > max_grad:
                t_eff = math.ceil(abs(area) / max_grad / system.grad_raster_time) * system.grad_raster_time
                amplitude2 = area / t_eff
                rise_time = math.ceil(
                    abs(amplitude2) / max_slew / system.grad_raster_time) * system.grad_raster_time

            flat_time = t_eff - rise_time
            fall_time = rise_time

    if abs(amplitude2) > max_grad:
        raise ValueError("Amplitude violation")

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

    return grad
