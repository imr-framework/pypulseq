import math
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.make_delay import make_delay
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts


def make_gauss_pulse(
    flip_angle: float,
    apodization: float = 0,
    bandwidth: float = 0,
    center_pos: float = 0.5,
    delay: float = 0,
    dwell: float = 0,
    duration: float = 4e-3,
    freq_offset: float = 0,
    max_grad: float = 0,
    max_slew: float = 0,
    phase_offset: float = 0,
    return_gz: bool = False,
    return_delay: bool = False,
    slice_thickness: float = 0,
    system: Opts = Opts(),
    time_bw_product: float = 4,
    use: str = str(),
) -> Union[
    SimpleNamespace,
    Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace],
    Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace],
]:
    """
    Create a [optionally slice selective] Gauss pulse.

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    apodization : float, default=0
        Apodization.
    bandwidth : float, default=0
        Bandwidth in Hertz (Hz).
    center_pos : float, default=0.5
        Position of peak.
    delay : float, default=0
        Delay in seconds (s).
    dwell : float, default=0
    duration : float, default=4e-3
        Duration in seconds (s).
    freq_offset : float, default=0
        Frequency offset in Hertz (Hz).
    max_grad : float, default=0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, default=0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, default=0
        Phase offset in Hertz (Hz).
    return_delay : bool, default=False
        Boolean flag to indicate if the delay event has to be returned.
    return_gz : bool, default=False
        Boolean flag to indicate if the slice-selective gradient has to be returned.
    slice_thickness : float, default=0
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    system : Opts, default=Opts()
        System limits.
    time_bw_product : int, default=4
        Time-bandwidth product.
    use : str, default=str()
        Use of radio-frequency gauss pulse event. Must be one of 'excitation', 'refocusing' or 'inversion'.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency gauss pulse event.
    gz : SimpleNamespace, optional
        Accompanying slice select trapezoidal gradient event.
    gzr : SimpleNamespace, optional
        Accompanying slice select rephasing trapezoidal gradient event.
    delay : SimpleNamespace, optional
        Delay event.

    Raises
    ------
    ValueError
        If invalid `use` is passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If `return_gz=True` and `slice_thickness` was not passed.
    """
    valid_use_pulses = ["excitation", "refocusing", "inversion"]
    if use != "" and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}"
        )

    if dwell == 0:
        dwell = system.rf_raster_time

    if bandwidth == 0:
        BW = time_bw_product / duration
    else:
        BW = bandwidth
    alpha = apodization
    N = int(np.round(duration / dwell))
    t = (np.arange(1, N + 1) - 0.5) * dwell
    tt = t - (duration * center_pos)
    window = 1 - alpha + alpha * np.cos(2 * np.pi * tt / duration)
    signal = window * __gauss(BW * tt)
    flip = np.sum(signal) * dwell * 2 * np.pi
    signal = signal * flip_angle / flip

    rf = SimpleNamespace()
    rf.type = "rf"
    rf.signal = signal
    rf.t = t
    rf.shape_dur = N * dwell
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay
    if use != "":
        rf.use = use

    if rf.dead_time > rf.delay:
        rf.delay = rf.dead_time

    if return_gz:
        if slice_thickness == 0:
            raise ValueError("Slice thickness must be provided")

        if max_grad > 0:
            system.max_grad = max_grad

        if max_slew > 0:
            system.max_slew = max_slew

        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(
            channel="z", system=system, flat_time=duration, flat_area=area
        )
        gzr = make_trapezoid(
            channel="z",
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz.area - area),
        )

        if rf.delay > gz.rise_time:
            gz.delay = (
                np.ceil((rf.delay - gz.rise_time) / system.grad_raster_time)
                * system.grad_raster_time
            )

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

    if rf.ringdown_time > 0 and return_delay:
        delay = make_delay(calc_duration(rf) + rf.ringdown_time)

    # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
    negative_zero_indices = np.where(rf.signal == -0.0)
    rf.signal[negative_zero_indices] = 0

    if return_gz and return_delay:
        return rf, gz, gzr, delay
    elif return_gz:
        return rf, gz, gzr
    else:
        return rf


def __gauss(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.pi * np.square(x))
