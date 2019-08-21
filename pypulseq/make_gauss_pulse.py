import math
from types import SimpleNamespace

import numpy as np

from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def make_gauss_pulse(flip_angle: float, system: Opts = Opts(), duration: float = 0, freq_offset: float = 0,
                     phase_offset: float = 0, time_bw_product: float = 4, bandwidth: float = 0, apodization: float = 0,
                     center_pos: float = 0.5, max_grad: float = 0, max_slew: float = 0, slice_thickness: float = 0,
                     delay: float = 0, use: str = None):
    """
    Creates a radio-frequency gauss pulse event and optionally accompanying slice select and slice select rephasing
    trapezoidal gradient events.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    duration : float, optional
        Duration in milliseconds (ms). Default is 0.
    freq_offset : float, optional
        Frequency offset in Hertz (Hz). Default is 0.
    phase_offset : float, optional
        Phase offset in Hertz (Hz). Default is 0.
    time_bw_product : int, optional
        Time-bandwidth product. Default is 4.
    bandwidth : float, optional
        Bandwidth in Hertz (Hz). Default is 0.
    apodization : float, optional
        Apodization. Default is 0.
    center_pos : float, optional
        Position of peak. Default is 0.5 (midway).
     max_grad : float, optional
        Maximum gradient strength of accompanying slice select trapezoidal event. Default is 0.
     max_slew : float, optional
        Maximum slew rate of accompanying slice select trapezoidal event. Default is 0.
    slice_thickness : float, optional
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event. Default is 0.
    delay : float, optional
        Delay in milliseconds (ms).
    use : str, optional
        Use of radio-frequency gauss pulse event. Must be one of 'excitation', 'refocusing' or 'inversion'.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency gauss pulse event.
    gz : SimpleNamespace
        Accompanying slice select trapezoidal gradient event.
    gzr : SimpleNamespace
        Accompanying slice select rephasing trapezoidal gradient event.
    """
    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use is not None and use not in valid_use_pulses:
        raise ValueError(
            f'Invalid use parameter. Must be one of excitation, refocusing or inversion. You passed: {use}')

    if bandwidth == 0:
        BW = time_bw_product / duration
    else:
        BW = bandwidth
    alpha = apodization
    N = int(round(duration / 1e-6))
    t = np.arange(1, N + 1) * system.rf_raster_time
    tt = t - (duration * center_pos)
    window = 1 - alpha + alpha * np.cos(2 * np.pi * tt / duration)
    signal = np.multiply(window, __gauss(BW * tt))
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
    if use is not None:
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


def __gauss(x):
    return np.exp(-np.pi * np.square(x))
