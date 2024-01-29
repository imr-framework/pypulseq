import math
from types import SimpleNamespace
from typing import Tuple, Union
from copy import copy

import numpy as np
import sigpy.mri.rf as rf
import sigpy.plot as pl

from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts


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
    system: Opts = None,
    time_bw_product: float = 4,
    pulse_cfg: SigpyPulseOpts = SigpyPulseOpts(),
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
    apodization : float, optional, default=0
        Apodization.
    center_pos : float, optional, default=0.5
        Position of peak.5 (midway).
    delay : float, optional, default=0
        Delay in seconds (s).
    duration : float, optional, default=4e-3
        Duration in seconds (s).
    freq_offset : float, optional, default=0
        Frequency offset in Hertz (Hz).
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
        System limits. Default is a system limits object initialised to default values.
    time_bw_product : float, optional, default=4
        Time-bandwidth product.
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
    if system == None:
        system = Opts.default
        
    valid_use_pulses = ["excitation", "refocusing", "inversion"]
    if use != "" and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}"
        )

    if pulse_cfg.pulse_type == "slr":
        [signal, t, pulse] = make_slr(
            flip_angle=flip_angle,
            time_bw_product=time_bw_product,
            duration=duration,
            system=system,
            pulse_cfg=pulse_cfg,
            disp=plot,
        )
    if pulse_cfg.pulse_type == "sms":
        [signal, t, pulse] = make_sms(
            flip_angle=flip_angle,
            time_bw_product=time_bw_product,
            duration=duration,
            system=system,
            pulse_cfg=pulse_cfg,
            disp=plot,
        )

    rfp = SimpleNamespace()
    rfp.type = "rf"
    rfp.signal = signal
    rfp.t = t
    rfp.shape_dur = t[-1]
    rfp.freq_offset = freq_offset
    rfp.phase_offset = phase_offset
    rfp.dead_time = system.rf_dead_time
    rfp.ringdown_time = system.rf_ringdown_time
    rfp.delay = delay

    if use != "":
        rfp.use = use

    if rfp.dead_time > rfp.delay:
        rfp.delay = rfp.dead_time

    if return_gz:
        if slice_thickness == 0:
            raise ValueError("Slice thickness must be provided")

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew
        BW = time_bw_product / duration
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

        if rfp.delay > gz.rise_time:
            gz.delay = (
                math.ceil((rfp.delay - gz.rise_time) / system.grad_raster_time)
                * system.grad_raster_time
            )

        if rfp.delay < (gz.rise_time + gz.delay):
            rfp.delay = gz.rise_time + gz.delay

    if rfp.ringdown_time > 0:
        t_fill = np.arange(1, round(rfp.ringdown_time / 1e-6) + 1) * 1e-6
        rfp.t = np.concatenate((rfp.t, rfp.t[-1] + t_fill))
        rfp.signal = np.concatenate((rfp.signal, np.zeros(len(t_fill))))

    # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
    negative_zero_indices = np.where(rfp.signal == -0.0)
    rfp.signal[negative_zero_indices] = 0

    if return_gz:
        return rfp, gz, gzr, pulse
    else:
        return rfp


def make_slr(
    flip_angle: float,
    time_bw_product: float = 4,
    duration: float = 0,
    system: Opts = None,
    pulse_cfg: SigpyPulseOpts = SigpyPulseOpts(),
    disp: bool = False,
):
    if system == None:
        system = Opts.default
        
    N = int(round(duration / 1e-6))
    t = np.arange(1, N + 1) * system.rf_raster_time

    # Insert sigpy
    ptype = pulse_cfg.ptype
    ftype = pulse_cfg.ftype
    d1 = pulse_cfg.d1
    d2 = pulse_cfg.d2
    cancel_alpha_phs = pulse_cfg.cancel_alpha_phs

    pulse = rf.slr.dzrf(
        n=N,
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
            np.arange(
                -20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000
            ),
            True,
        )
        Mxy = 2 * np.multiply(np.conj(a), b)
        pl.LinePlot(Mxy)

    return signal, t, pulse


def make_sms(
    flip_angle: float,
    time_bw_product: float = 4,
    duration: float = 0,
    system: Opts = None,
    pulse_cfg: SigpyPulseOpts = SigpyPulseOpts(),
    disp: bool = False,
):
    if system == None:
        system = Opts.default
        
    N = int(round(duration / 1e-6))
    t = np.arange(1, N + 1) * system.rf_raster_time

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
        n=N,
        tb=time_bw_product,
        ptype=ptype,
        ftype=ftype,
        d1=d1,
        d2=d2,
        cancel_alpha_phs=cancel_alpha_phs,
    )
    pulse = rf.multiband.mb_rf(
        pulse_in, n_bands=n_bands, band_sep=band_sep, phs_0_pt=phs_0_pt
    )

    flip = np.sum(pulse) * system.rf_raster_time * 2 * np.pi
    signal = pulse * flip_angle / flip

    if disp:
        pl.LinePlot(pulse_in)
        pl.LinePlot(pulse)
        pl.LinePlot(signal)
        # Simulate it
        [a, b] = rf.sim.abrm(
            pulse,
            np.arange(
                -20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000
            ),
            True,
        )
        Mxy = 2 * np.multiply(np.conj(a), b)
        pl.LinePlot(Mxy)

    return signal, t, pulse
