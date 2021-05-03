import math
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def make_arbitrary_rf(signal: np.ndarray, flip_angle: float, bandwidth: float = 0, delay: float = 0,
                      freq_offset: float = 0, max_grad: float = 0, max_slew: float = 0, phase_offset: float = 0,
                      return_gz: bool = False, slice_thickness: float = 0, system: Opts = Opts(),
                      time_bw_product: float = 0,
                      use: str = str()) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace]]:
    """
    Creates a radio-frequency pulse event with arbitrary pulse shape and optionally an accompanying slice select
    trapezoidal gradient event.

    Parameters
    ----------
    signal : numpy.ndarray
        Arbitrary waveform.
    flip_angle : float
        Flip angle in radians.
    bandwidth : float, default=0
        Bandwidth in Hertz (Hz).
    delay : float, default=0
        Delay in milliseconds (ms) of accompanying slice select trapezoidal event.
    freq_offset : float, default=0
        Frequency offset in Hertz (Hz).
    max_grad : float, default=system.max_grad
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, default=system.max_slew
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, default=0
        Phase offset in Hertz (Hz).a
    return_gz : bool, default=False
        Boolean flag to indicate if slice-selective gradient has to be returned.
    slice_thickness : float, default=0
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    system : Opts, default=Opts()
        System limits.
    time_bw_product : float, default=4
        Time-bandwidth product.
    use : str, default=str()
        Use of arbitrary radio-frequency pulse event. Must be one of 'excitation', 'refocusing' or 'inversion'.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency pulse event with arbitrary pulse shape.
    gz : SimpleNamespace, if return_gz=True
        Slice select trapezoidal gradient event accompanying the arbitrary radio-frequency pulse event.

    Raises
    ------
    ValueError
        If invalid `use` parameter is passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If `signal` with ndim > 1 is passed.
        If `return_gz=True`, and `slice_thickness` and `bandwith` are not passed.
    """
    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use != '' and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}")

    signal = np.squeeze(signal)
    if signal.ndim > 1:
        raise ValueError(f'signal should have ndim=1. Passed ndim={signal.ndim}')
    signal = signal / bp.abs(np.sum(signal * system.rf_raster_time)) * flip_angle / (2 * np.pi)

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

    if use != '':
        rf.use = use

    if rf.dead_time > rf.delay:
        rf.delay = rf.dead_time

    if return_gz:
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

    if rf.ringdown_time > 0:
        t_fill = np.arange(1, round(rf.ringdown_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, rf.t[-1] + t_fill))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    return rf, gz if return_gz else rf
