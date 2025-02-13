import math
from copy import copy
from types import SimpleNamespace
from typing import Tuple, Union
from warnings import warn

import numpy as np

try:
    import sigpy.mri.rf as rf
    import sigpy.plot as pl
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "SigPy is not installed. Install it using 'pip install sigpy' or 'pip install pypulseq[sigpy]'."
    ) from err

from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq.utils.tracing import trace, trace_enabled


def sigpy_n_seq(
    flip_angle: float,
    delay: float = 0,
    duration: float = 4e-3,
    freq_offset: float = 0,
    center_pos: float = 0.5,
    max_grad: float = 0,
    max_slew: float = 0,
    phase_offset: float = 0,
    return_gz: bool = True,
    slice_thickness: float = 0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 4,
    pulse_cfg: Union[SigpyPulseOpts, None] = None,
    use: str = str(),
    plot: bool = True,
) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]]:
    """
    Creates a radio-frequency sinc pulse event using the sigpy rf pulse library and optionally accompanying slice select, slice select rephasing
    trapezoidal gradient events.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    delay : float, optional, default=0
        Delay in seconds (s).
    duration : float, optional, default=4e-3
        Duration in seconds (s).
    freq_offset : float, optional, default=0
        Frequency offset in Hertz (Hz).
    center_pos : float, optional, default=0.5
        Position of peak.5 (midway).
    max_grad : float, optional, default=0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, optional, default=0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, optional, default=0
        Phase offset in Hertz (Hz).
    return_gz:bool, default=False
        Boolean flag to indicate if slice-selective gradient has to be returned.
    slice_thickness : float, optional, default=0
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    system : Opts, optional
        System limits. Default is a system limits object initialized to default values.
    time_bw_product : float, optional, default=4
        Time-bandwidth product.
    pulse_cfg: SigpyPulseOpts, optional, default=None
        Pulse configuration options. Possible keys are:
        - pulse_type: str, optional, default='slr'
            Pulse type. Must be one of 'slr' or 'sms'.
        - ptype: str, optional, default='st'
            Pulse design method. Must be one of 'st', 'ex', 'inv', 'sat', 'se', 'fi', 'fs', 'se'.
        - ftype: str, optional, default='ls'
            Filter type. Must be one of 'ls', 'pm', 'min', 'max', 'ap'.
        - d1: float, optional, default=0.01
            Passband ripple.
        - d2: float, optional, default=0.01
            Stopband ripple.
        - cancel_alpha_phs: bool, optional, default=False
            Cancel alpha phase.
        - n_bands: int, optional, default=3
            Number of bands. SMS only.
        - band_sep: float, optional, default=20
            Band separation. SMS only.
        - phs_0_pt: str, optional, default='None'
            Phase 0 point. SMS only.
    use : str, optional, default=str()
        Use of radio-frequency sinc pulse. Must be one of 'excitation', 'refocusing' or 'inversion'.
    plot: bool, optional, default=True
        Show sigpy plot outputs

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency sinc pulse event.
    gz : SimpleNamespace, optional
        Accompanying slice select trapezoidal gradient event. Returned only if `slice_thickness` is provided.
    gzr : SimpleNamespace, optional
        Accompanying slice select rephasing trapezoidal gradient event. Returned only if `slice_thickness` is provided.

    Raises
    ------
    ValueError
        If invalid `use` parameter was passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If `return_gz=True` and `slice_thickness` was not provided.
    """
    if system is None:
        system = Opts.default

    if pulse_cfg is None:
        pulse_cfg = SigpyPulseOpts()

    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use != '' and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}"
        )

    if pulse_cfg.pulse_type == 'slr':
        [signal, t, pulse] = make_slr(
            flip_angle=flip_angle,
            time_bw_product=time_bw_product,
            duration=duration,
            system=system,
            pulse_cfg=pulse_cfg,
            disp=plot,
        )
    if pulse_cfg.pulse_type == 'sms':
        [signal, t, pulse] = make_sms(
            flip_angle=flip_angle,
            time_bw_product=time_bw_product,
            duration=duration,
            system=system,
            pulse_cfg=pulse_cfg,
            disp=plot,
        )

    rfp = SimpleNamespace()
    rfp.type = 'rf'
    rfp.signal = signal
    rfp.t = t
    rfp.shape_dur = t[-1]
    rfp.freq_offset = freq_offset
    rfp.phase_offset = phase_offset
    rfp.dead_time = system.rf_dead_time
    rfp.ringdown_time = system.rf_ringdown_time
    rfp.delay = delay

    if use != '':
        rfp.use = use

    if rfp.dead_time > rfp.delay:
        warn(
            f'Specified RF delay {rfp.delay * 1e6:.2f} us is less than the dead time {rfp.dead_time * 1e6:.0f} us. Delay was increased to the dead time.',
            stacklevel=2,
        )
        rfp.delay = rfp.dead_time

    if return_gz:
        if slice_thickness == 0:
            raise ValueError('Slice thickness must be provided')

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew
        bandwidth = time_bw_product / duration
        amplitude = bandwidth / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)
        gzr = make_trapezoid(
            channel='z',
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz.area - area),
        )

        if rfp.delay > gz.rise_time:
            gz.delay = math.ceil((rfp.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rfp.delay < (gz.rise_time + gz.delay):
            rfp.delay = gz.rise_time + gz.delay

    if rfp.ringdown_time > 0:
        t_fill = np.arange(1, round(rfp.ringdown_time / 1e-6) + 1) * 1e-6
        rfp.t = np.concatenate((rfp.t, rfp.t[-1] + t_fill))
        rfp.signal = np.concatenate((rfp.signal, np.zeros(len(t_fill))))

    # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
    negative_zero_indices = np.where(rfp.signal == -0.0)
    rfp.signal[negative_zero_indices] = 0

    if trace_enabled():
        rfp.trace = trace()

    if return_gz:
        return rfp, gz, gzr
    else:
        return rfp


def make_slr(
    flip_angle: float,
    time_bw_product: float = 4,
    duration: float = 0,
    system: Union[Opts, None] = None,
    pulse_cfg: Union[SigpyPulseOpts, None] = None,
    disp: bool = False,
):
    if system is None:
        system = Opts.default

    if pulse_cfg is None:
        pulse_cfg = SigpyPulseOpts()

    n_samples = int(round(duration / 1e-6))
    t = np.arange(1, n_samples + 1) * system.rf_raster_time

    # Insert sigpy
    ptype = pulse_cfg.ptype
    ftype = pulse_cfg.ftype
    d1 = pulse_cfg.d1
    d2 = pulse_cfg.d2
    cancel_alpha_phs = pulse_cfg.cancel_alpha_phs

    pulse = rf.slr.dzrf(
        n=n_samples,
        tb=time_bw_product,
        ptype=ptype,
        ftype=ftype,
        d1=d1,
        d2=d2,
        cancel_alpha_phs=cancel_alpha_phs,
    )
    flip = np.sum(pulse) * system.rf_raster_time * 2 * np.pi
    signal = pulse * flip_angle / flip

    if disp:
        pl.LinePlot(pulse)
        pl.LinePlot(signal)

        # Simulate it
        [a, b] = rf.sim.abrm(
            pulse,
            np.arange(-20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000),
            True,
        )
        mag_xy = 2 * np.multiply(np.conj(a), b)
        pl.LinePlot(mag_xy)

    return signal, t, pulse


def make_sms(
    flip_angle: float,
    time_bw_product: float = 4,
    duration: float = 0,
    system: Union[Opts, None] = None,
    pulse_cfg: Union[SigpyPulseOpts, None] = None,
    disp: bool = False,
):
    if system is None:
        system = Opts.default

    if pulse_cfg is None:
        pulse_cfg = SigpyPulseOpts()

    n_samples = int(round(duration / 1e-6))
    t = np.arange(1, n_samples + 1) * system.rf_raster_time

    # Insert sigpy
    ptype = pulse_cfg.ptype
    ftype = pulse_cfg.ftype
    d1 = pulse_cfg.d1
    d2 = pulse_cfg.d2
    cancel_alpha_phs = pulse_cfg.cancel_alpha_phs
    n_bands = pulse_cfg.n_bands
    band_sep = pulse_cfg.band_sep
    phs_0_pt = pulse_cfg.phs_0_pt

    pulse_in = rf.slr.dzrf(
        n=n_samples,
        tb=time_bw_product,
        ptype=ptype,
        ftype=ftype,
        d1=d1,
        d2=d2,
        cancel_alpha_phs=cancel_alpha_phs,
    )
    pulse = rf.multiband.mb_rf(pulse_in, n_bands=n_bands, band_sep=band_sep, phs_0_pt=phs_0_pt)

    flip = np.sum(pulse) * system.rf_raster_time * 2 * np.pi
    signal = pulse * flip_angle / flip

    if disp:
        pl.LinePlot(pulse_in)
        pl.LinePlot(pulse)
        pl.LinePlot(signal)
        # Simulate it
        [a, b] = rf.sim.abrm(
            pulse,
            np.arange(-20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000),
            True,
        )
        mag_xy = 2 * np.multiply(np.conj(a), b)
        pl.LinePlot(mag_xy)

    return signal, t, pulse
