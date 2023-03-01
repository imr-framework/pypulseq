from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.make_delay import make_delay
from pypulseq.opts import Opts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses


def make_block_pulse(
    flip_angle: float,
    bandwidth: float = 0,
    delay: float = 0,
    duration: float = 4e-3,
    freq_offset: float = 0,
    phase_offset: float = 0,
    return_delay: bool = False,
    system: Opts = Opts(),
    time_bw_product: float = 0,
    use: str = str(),
) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace]]:
    """
    Create a block pulse with optional slice selectiveness.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    bandwidth : float, default=0
        Bandwidth in Hertz (hz).
    delay : float, default=0
        Delay in seconds (s) of accompanying slice select trapezoidal event.
    duration : float, default=4e-3
        Duration in seconds (s).
    freq_offset : float, default=0
        Frequency offset in Hertz (Hz).
    phase_offset : float, default=0
        Phase offset Hertz (Hz).
    return_delay : bool, default=False
        Boolean flag to indicate if the delay event has to be returned.
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
    delay : SimpleNamespace, optional
        Slice select trapezoidal gradient event accompanying the radio-frequency block pulse event.

    Raises
    ------
    ValueError
        If invalid `use` parameter is passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If neither `bandwidth` nor `duration` are passed.
        If `return_gz=True`, and `slice_thickness` is not passed.
    """
    valid_use_pulses = get_supported_rf_uses()
    if use != "" and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}"
        )

    if duration == 0:
        if time_bw_product > 0:
            duration = time_bw_product / bandwidth
        elif bandwidth > 0:
            duration = 1 / (4 * bandwidth)
        else:
            raise ValueError("Either bandwidth or duration must be defined")

    BW = 1 / (4 * duration)
    N = np.round(duration / system.rf_raster_time)
    t = np.array([0, N]) * system.rf_raster_time
    signal = flip_angle / (2 * np.pi) / duration * np.ones_like(t)

    rf = SimpleNamespace()
    rf.type = "rf"
    rf.signal = signal
    rf.t = t
    rf.shape_dur = t[-1]
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay

    if use != "":
        rf.use = use

    if rf.dead_time > rf.delay:
        rf.delay = rf.dead_time

    if rf.ringdown_time > 0 and return_delay:
        delay = make_delay(calc_duration(rf) + rf.ringdown_time)

    if return_delay:
        return rf, delay
    else:
        return rf
