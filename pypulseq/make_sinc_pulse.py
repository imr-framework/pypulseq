import math
from types import SimpleNamespace

import numpy as np

from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def make_sinc_pulse(flip_angle, system=Opts(), duration=0, freq_offset=0, phase_offset=0, time_bw_product=4,
                    apodization=0, center_pos=0.5, max_grad=0, max_slew=0, slice_thickness=0, delay=0, use=''):
    """
    Makes a Holder object for an RF pulse Event.

    Parameters
    ----------
    kwargs : dict
        Key value mappings of RF Event parameters_params and values.
    nargout: int
        Number of output arguments to be returned. Default is 1, only RF Event is returned. Passing any number greater
        than 1 will return the Gz Event along with the RF Event.

    Returns
    -------
    rf : Holder
        RF Event configured based on supplied kwargs.
    gz : Holder
        Slice select trapezoidal gradient Event.
    """

    BW = time_bw_product / duration
    alpha = apodization
    N = int(round(duration / 1e-6))
    t = np.arange(1, N + 1) * system.rf_raster_time
    tt = t - (duration * center_pos)
    window = 1 - alpha + alpha * np.cos(2 * np.pi * tt / duration)
    signal = np.multiply(window, np.sinc(BW * tt))
    flip = np.sum(signal) * system.rf_raster_time * 2 * np.pi
    signal = signal * flip_angle / flip

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay
    if use != '':
        rf.use = use

    if rf.dead_time > rf.delay:
        rf.delay = rf.dead_time

    try:
        if slice_thickness == 0:
            raise ValueError('Slice thickness must be provided')

        if max_grad > 0:
            system.max_grad = max_grad

        if max_slew > 0:
            system.max_slew = max_slew

        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)
        gzr = make_trapezoid(channel='z', system=system, area=-area * (1 - center_pos) - 0.5 * (gz.area - area))

        if rf.delay > gz.rise_time:
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay
    except:
        gz = None
        gzr = None

    if rf.ringdown_time > 0:
        t_fill = np.arange(1, round(rf.ringdown_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, rf.t[-1] + t_fill))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
    negative_zero_indices = np.where(rf.signal == -0.0)
    rf.signal[negative_zero_indices] = 0

    return rf, gz, gzr
