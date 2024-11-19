from types import SimpleNamespace
from typing import Tuple, Union
from copy import copy

import numpy as np
import math
from warnings import warn

from pypulseq import make_delay, calc_duration
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses
from pypulseq.utils.tracing import trace_enabled, trace


def make_arbitrary_rf(
    signal: np.ndarray,
    flip_angle: float,
    bandwidth: float = 0,
    delay: float = 0,
    dwell: float = 0,
    freq_offset: float = 0,
    no_signal_scaling: bool = False,
    max_grad: float = 0,
    max_slew: float = 0,
    phase_offset: float = 0,
    return_delay: bool = False,
    return_gz: bool = False,
    slice_thickness: float = 0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 0,
    use: str = str(),
) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace]]:
    """
    Create an RF pulse with the given pulse shape.

    Parameters
    ----------
    signal : numpy.ndarray
        Arbitrary waveform.
    flip_angle : float
        Flip angle in radians.
    bandwidth : float, default=0
        Bandwidth in Hertz (Hz).
    delay : float, default=0
        Delay in seconds (s) of accompanying slice select trapezoidal event.
    dwell : float, default=0
        Temporal sampling step of waveform. If set to 0, will use `system.rf_raster_time`.
    freq_offset : float, default=0
        Frequency offset in Hertz (Hz).
    no_signal_scaling : bool, default=False
        If set to True no rescaling of the RF amplitude will happen. E.g. for adiabatic pulses.
    max_grad : float, default=system.max_grad
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, default=system.max_slew
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, default=0
        Phase offset in Hertz (Hz).a
    return_delay : bool, default=False
        Boolean flag to indicate if delay has to be returned.
    return_gz : bool, default=False
        Boolean flag to indicate if slice-selective gradient has to be returned.
    slice_thickness : float, default=0
        Slice thickness (m) of accompanying slice select trapezoidal event. The slice thickness determines the area of the
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
    gz : SimpleNamespace, optional
        Slice select trapezoidal gradient event accompanying the arbitrary radio-frequency pulse event.

    Raises
    ------
    ValueError
        If invalid `use` parameter is passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If `signal` with ndim > 1 is passed.
        If `return_gz=True`, and `slice_thickness` and `bandwidth` are not passed.
    """
    if system is None:
        system = Opts.default

    valid_use_pulses = get_supported_rf_uses()
    if use != '' and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}"
        )

    if dwell == 0:
        dwell = system.rf_raster_time

    signal = np.squeeze(signal)
    if signal.ndim > 1:
        raise ValueError(f'signal should have ndim=1. Passed ndim={signal.ndim}')

    if not no_signal_scaling:
        signal = signal / np.abs(np.sum(signal * dwell)) * flip_angle / (2 * np.pi)

    N = len(signal)
    duration = N * dwell
    t = (np.arange(1, N + 1) - 0.5) * dwell

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.shape_dur = duration
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay

    if use != '':
        rf.use = use

    if rf.dead_time > rf.delay:
        warn(
            f'Specified RF delay {rf.delay*1e6:.2f} us is less than the dead time {rf.dead_time*1e6:.0f} us. Delay was increased to the dead time.',
            stacklevel=2,
        )
        rf.delay = rf.dead_time

    if return_gz:
        if slice_thickness <= 0:
            raise ValueError('Slice thickness must be provided.')
        if bandwidth <= 0:
            raise ValueError('Bandwidth of pulse must be provided.')

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad
        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew

        BW = bandwidth
        if time_bw_product > 0:
            BW = time_bw_product / duration

        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)

        if rf.delay > gz.rise_time:
            # Round-up to gradient raster
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

    if rf.ringdown_time > 0 and return_delay:
        delay = make_delay(calc_duration(rf) + rf.ringdown_time)

    if trace_enabled():
        rf.trace = trace()

    if return_gz and return_delay:
        return rf, gz, delay
    elif return_gz:
        return rf, gz
    else:
        return rf
