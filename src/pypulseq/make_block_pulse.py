from types import SimpleNamespace
from typing import Union
from warnings import warn

import numpy as np

from pypulseq.opts import Opts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses
from pypulseq.utils.tracing import trace, trace_enabled


def make_block_pulse(
    flip_angle: float,
    delay: float = 0,
    duration: Union[float, None] = None,
    bandwidth: Union[float, None] = None,
    time_bw_product: Union[float, None] = None,
    freq_offset: float = 0,
    phase_offset: float = 0,
    system: Union[Opts, None] = None,
    use: str = str(),
) -> SimpleNamespace:
    """
    Create a block (RECT or hard) pulse.

    Define duration, or bandwidth, or bandwidth and time_bw_product.
    If none are provided a default 4 ms pulse will be generated.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    delay : float, default=0
        Delay in seconds (s).
    duration : float, default=None
        Duration in seconds (s).
    bandwidth : float, default=None
        Bandwidth in Hertz (Hz).
        If supplied without time_bw_product duration = 1 / (4 * bandwidth)
    time_bw_product : float, default=None
        Time-bandwidth product.
        If supplied with bandwidth, duration = time_bw_product / bandwidth
    freq_offset : float, default=0
        Frequency offset in Hertz (Hz).
    phase_offset : float, default=0
        Phase offset Hertz (Hz).
    system : Opts, default=Opts()
        System limits.
    use : str, default=str()
        Use of radio-frequency block pulse event.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency block pulse event.

    Raises
    ------
    ValueError
        If invalid `use` parameter is passed.
        One of bandwidth or duration must be defined, but not both.
        One of bandwidth or duration must be defined and be > 0.
    """
    if system is None:
        system = Opts.default

    valid_use_pulses = get_supported_rf_uses()
    if use != '' and use not in valid_use_pulses:
        raise ValueError(f'Invalid use parameter. Must be one of {valid_use_pulses}. Passed: {use}')

    if duration is None and bandwidth is None:
        warn('Using default 4 ms duration for block pulse.')
        duration = 4e-3
    elif duration is not None and bandwidth is not None and duration > 0:
        # Multiple arguments
        raise ValueError('One of bandwidth or duration must be defined, but not both.')
    elif duration is not None and duration > 0:
        # Explicitly handle this most expected case.
        # There is probably a better way of writing this if block
        pass
    elif duration is None and bandwidth is not None and bandwidth > 0:
        if time_bw_product is not None and time_bw_product > 0:
            duration = time_bw_product / bandwidth
        else:
            duration = 1 / (4 * bandwidth)
    else:
        # Invalid arguments
        raise ValueError(
            'One of bandwidth or duration must be defined and be > 0. '
            f'duration = {duration} s, bandwidth = {bandwidth} Hz.'
        )

    n_samples = round(duration / system.rf_raster_time)
    t = np.array([0, n_samples]) * system.rf_raster_time
    signal = flip_angle / (2 * np.pi) / duration * np.ones_like(t)

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.shape_dur = t[-1]
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay

    if use != '':
        rf.use = use

    if rf.dead_time > rf.delay:
        warn(
            f'Specified RF delay {rf.delay * 1e6:.2f} us is less than the dead time {rf.dead_time * 1e6:.0f} us. Delay was increased to the dead time.',
            stacklevel=2,
        )
        rf.delay = rf.dead_time

    if trace_enabled():
        rf.trace = trace()

    return rf
