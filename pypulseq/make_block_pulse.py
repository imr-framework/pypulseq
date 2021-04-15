import math
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def make_block_pulse(flip_angle: float, bandwidth: float = 0, delay: float = 0, duration: float = 0,
                     freq_offset: float = 0, max_grad: float = 0, max_slew: float = 0, phase_offset: float = 0,
                     return_gz: bool = False, system: Opts = Opts(), slice_thickness: float = 0,
                     time_bw_product: float = 0,
                     use: str = str()) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace]]:
    """
    Creates a radio-frequency block pulse event and optionally an accompanying slice select trapezoidal gradient event.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    bandwidth : float, default=0
        Bandwidth in Hertz (hz).
    delay : float, default=0
        Delay in milliseconds (ms) of accompanying slice select trapezoidal event.
    duration : float, default=0
        Duration in milliseconds (ms).
    freq_offset : float, default=0
        Frequency offset in Hertz (Hz).
    max_grad : float, default=0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, default=0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, default=0
        Phase offset Hertz (Hz).
    return_gz : bool, default=False
        Boolean flag to indicate if slice-selective gradient has to be returned.
    slice_thickness : float, default=0
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    system : Opts, default=Opts()
        System limits.
    time_bw_product : float, default=0
        Time-bandwidth product.
    use : str, default=str()
        Use of radio-frequency block pulse event. Must be one of 'excitation', 'refocusing' or 'inversion'.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency block pulse event.
    gz : SimpleNamespace
        Slice select trapezoidal gradient event accompanying the radio-frequency block pulse event.

    Raises
    ------
    ValueError
        If invalid `use` parameter is passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If neither `bandwidth` nor `duration` are passed.
        If `return_gz=True`, and `slice_thickness` is not passed.
    """
    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use != '' and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}")

    if duration == 0:
        if time_bw_product > 0:
            duration = time_bw_product / bandwidth
        elif bandwidth > 0:
            duration = 1 / (4 * bandwidth)
        else:
            raise ValueError('Either bandwidth or duration must be defined')

    BW = 1 / (4 * duration)
    N = round(duration / 1e-6)
    t = np.arange(1, N + 1) * system.rf_raster_time
    signal = flip_angle / (2 * np.pi) / duration * np.ones(len(t))

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

    if return_gz:
        if slice_thickness < 0:
            raise ValueError('Slice thickness must be provided')

        if max_grad > 0:
            system.max_grad = max_grad
        if max_slew > 0:
            system.max_slew = max_slew

        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)

        if rf.delay > gz.rise_time:
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

    if rf.ringdown_time > 0:
        t_fill = np.arange(1, round(rf.ringdown_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, (rf.t[-1] + t_fill)))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    if return_gz:
        return rf, gz
    else:
        return rf
