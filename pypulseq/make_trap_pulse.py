import math
from types import SimpleNamespace

from pypulseq.opts import Opts


def make_trapezoid(channel='z', system=Opts(), duration=0, area=-1, flat_time=None, flat_area=None, amplitude=None,
                   max_grad=0, max_slew=0, rise_time=0, delay=0):
    """
    Makes a Holder object for an trapezoidal gradient Event.

    Parameters
    ----------
    kwargs : dict
        Key value mappings of trapezoidal gradient Event parameters_params and values.

    Returns
    -------
    grad : Holder
        Trapezoidal gradient Event configured based on supplied kwargs.
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
        if amplitude is None:
            amplitude_temp = flat_area / flat_time

        if rise_time is None:
            rise_time = abs(amplitude_temp) / max_slew
            rise_time = math.ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
        fall_time, flat_time = rise_time, flat_time
    elif duration > 0:
        if amplitude is None:
            if rise_time is None:
                dC = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
                possible = duration ** 2 > 4 * abs(area) * dC
                amplitude_temp = (duration - math.sqrt(duration ** 2 - 4 * abs(area) * dC)) / (2 * dC)
            else:
                amplitude_temp = area / (duration - rise_time)
                possible = duration > 2 * rise_time and abs(amplitude_temp) < max_grad

            if not possible:
                raise ValueError('Requested area is too large for this gradient')

        if rise_time is None:
            rise_time = math.ceil(
                abs(amplitude_temp) / max_slew / system.grad_raster_time) * system.grad_raster_time

        fall_time = rise_time
        flat_time = duration - rise_time - fall_time

        if amplitude is None:
            amplitude_temp = area / (rise_time / 2 + fall_time / 2 + flat_time)

    else:
        if area is None:
            raise ValueError('Must supply a duration')
        else:
            rise_time = math.ceil(math.sqrt(abs(area) / max_slew) / system.grad_raster_time) * system.grad_raster_time
            amplitude_temp = area / rise_time
            t_eff = rise_time

            if abs(amplitude_temp) > max_grad:
                t_eff = math.ceil(abs(area) / max_grad / system.grad_raster_time) * system.grad_raster_time
                amplitude_temp = area / t_eff
                rise_time = math.ceil(
                    abs(amplitude_temp) / max_slew / system.grad_raster_time) * system.grad_raster_time

            flat_time = t_eff - rise_time
            fall_time = rise_time

    if abs(amplitude_temp) > max_grad:
        raise ValueError("Amplitude violation")

    grad = SimpleNamespace()
    grad.type = 'trap'
    grad.channel = channel
    grad.amplitude = amplitude_temp
    grad.rise_time = rise_time
    grad.flat_time = flat_time
    grad.fall_time = fall_time
    grad.area = amplitude_temp * (flat_time + rise_time / 2 + fall_time / 2)
    grad.flat_area = amplitude_temp * flat_time
    grad.delay = delay
    grad.first = 0
    grad.last = 0

    return grad
