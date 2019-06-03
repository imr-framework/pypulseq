import math
from types import SimpleNamespace

import numpy as np

from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def make_block_pulse(flip_angle, system=Opts(), duration=0, freq_offset=0, phase_offset=0, time_bw_product=0,
                     bandwidth=0, max_grad=0, max_slew=0, slice_thickness=0, delay=0, use=''):
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
    Tuple consisting of:
    rf : Holder
        RF Event configured based on supplied kwargs.
    gz : Holder
        Slice select trapezoidal gradient Event.
    """

    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use not in valid_use_pulses:
        raise Exception()

    if duration == 0:
        if time_bw_product > 0:
            duration = time_bw_product / bandwidth
        elif bandwidth > 0:
            duration = 1 / (4 * bandwidth)
        else:
            raise ValueError('Either bandwidth or duration must be defined')

    BW = 1 / (4 * duration)
    N = round(duration / 1e-6)
    t = np.arange(N) * system.rf_raster_time
    signal = flip_angle / (2 * np.pi) / duration * np.ones(len(t))

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ring_down_time = system.rf_ring_down_time

    if use != '':
        rf.use = use

    if rf.dead_time > rf.delay:
        rf.delay = rf.dead_time

    try:
        if slice_thickness < 0:
            raise ValueError('Slice thickness must be provided')

        if max_grad > 0:
            system.max_grad = max_grad
        if max_slew > 0:
            system.max_slew = max_slew

        amplitude = BW / slice_thickness
        area = amplitude * duration
        kwargs_for_trap = {'channel': 'z', 'system': system, 'flat_time': duration, 'flat_area': area}
        gz = make_trapezoid(kwargs_for_trap)

        if rf.delay > gz.rise_time:
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay
    except:
        gz = None

    if rf.ring_down_time > 0:
        t_fill = np.arange(round(rf.ring_down_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, (rf.t[-1] + t_fill)))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    return rf, gz
