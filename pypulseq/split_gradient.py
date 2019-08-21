from types import SimpleNamespace

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts


def split_gradient(grad: np.ndarray, system: Opts = Opts()):
    """
    Split gradient waveform `grad` into two gradient waveforms at the center.

    Parameters
    ----------
    grad : numpy.ndarray
        Gradient waveform to be split into two gradient waveforms.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    Returns
    -------
    grad1, grad2 : numpy.ndarray
        Split gradient waveforms.
    """
    grad_raster_time = system.grad_raster_time
    total_length = calc_duration(grad)

    if grad.type == 'trap':
        ch = grad.channel
        grad.delay = round(grad.delay / grad_raster_time) * grad_raster_time
        grad.rise_time = round(grad.rise_time / grad_raster_time) * grad_raster_time
        grad.flat_time = round(grad.flat_time / grad_raster_time) * grad_raster_time
        grad.fall_time = round(grad.fall_time / grad_raster_time) * grad_raster_time

        times = [0, grad.rise_time]
        amplitudes = [0, grad.amplitude]
        ramp_up = make_extended_trapezoid(channel=ch, system=system, times=times, amplitudes=amplitudes,
                                          skip_check=True)
        ramp_up.delay = grad.delay

        times = [0, grad.fall_time]
        amplitudes = [grad.amplitude, 0]
        ramp_down = make_extended_trapezoid(channel=ch, system=system, times=times, amplitudes=amplitudes,
                                            skip_check=True)
        ramp_down.delay = total_length - grad.fall_time
        ramp_down.t = ramp_down.t * grad_raster_time

        flat_top = SimpleNamespace()
        flat_top.type = 'grad'
        flat_top.channel = ch
        flat_top.delay = grad.delay + grad.rise_time
        flat_top.t = np.arange(step=grad_raster_time,
                               stop=ramp_down.delay - grad_raster_time - grad.delay - grad.rise_time)
        flat_top.waveform = grad.amplitude * np.ones(len(flat_top.t))
        flat_top.first = grad.amplitude
        flat_top.last = grad.amplitude

        return ramp_up, flat_top, ramp_down
    elif grad.type == 'grad':
        raise ValueError('Splitting of arbitrary gradients is not implemented yet.')
    else:
        raise ValueError('Splitting of unsupported event.')
