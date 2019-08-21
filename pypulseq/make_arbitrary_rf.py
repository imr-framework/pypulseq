from types import SimpleNamespace

import numpy as np
import math

from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
from types import SimpleNamespace


def make_arbitrary_rf(signal: np.ndarray, flip_angle: float, system: Opts = Opts(), freq_offset: float = 0,
                      phase_offset: float = 0, time_bw_product: float = 0, bandwidth: float = 0, max_grad: float = 0,
                      max_slew: float = 0, slice_thickness: float = 0, delay: float = 0,
                      use: str = None) -> SimpleNamespace:
    """
    Creates a radio-frequency pulse event with arbitrary pulse shape and optionally an accompanying slice select
    trapezoidal gradient event.

    Parameters
    ----------
    signal : np.ndarray
        Arbitrary waveform.
    flip_angle : float
        Flip angle in radians.
    system : Opts
        System limits. Default is a system limits object initialised to default values.
    freq_offset : float
        Frequency offset in Hertz (Hz). Default is 0.
    phase_offset : float
        Phase offset in Hertz (Hz). Default is 0.
    time_bw_product : float
        Time-bandwidth product. Default is 4.
    bandwidth : float
        Bandwidth in Hertz (Hz).
     max_grad : float
        Maximum gradient strength of accompanying slice select trapezoidal event. Default is `system`'s `max_grad`.
     max_slew : float
        Maximum slew rate of accompanying slice select trapezoidal event. Default is `system`'s `max_slew`.
    slice_thickness : float
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    delay : float
        Delay in milliseconds (ms) of accompanying slice select trapezoidal event.
    use : str
        Use of arbitrary radio-frequency pulse event. Must be one of 'excitation', 'refocusing' or 'inversion'.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency pulse event with arbitrary pulse shape.
    gz : SimpleNamespace
        Slice select trapezoidal gradient event accompanying the arbitrary radio-frequency pulse event.
    """
    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use is not None and use not in valid_use_pulses:
        raise ValueError(
            f'Invalid use parameter. Must be one of excitation, refocusing or inversion. You passed: {use}')

    signal = signal / np.sum(signal * system.rf_raster_time) * flip_angle / (2 * np.pi)

    N = len(signal)
    duration = N * system.rf_raster_time
    t = np.arange(1, N + 1) * system.rf_raster_time

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
        if slice_thickness <= 0:
            raise ValueError('Slice thickness must be provided.')
        if bandwidth <= 0:
            raise ValueError('Bandwidth of pulse must be provided.')

        if max_grad > 0:
            system.max_grad = max_grad
        if max_slew > 0:
            system.max_slew = max_slew

        BW = bandwidth
        if time_bw_product > 0:
            BW = time_bw_product / duration

        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)

        if rf.delay > gz.rise_time:
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay
    except:
        gz = None

    if rf.ringdown_time > 0:
        t_fill = np.arange(1, round(rf.ringdown_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, rf.t[-1] + t_fill))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    return rf, gz
