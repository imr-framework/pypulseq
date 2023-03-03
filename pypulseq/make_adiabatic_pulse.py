from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
from sigpy.mri.rf import hypsec, wurst

from pypulseq import eps
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_delay import make_delay
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses


def make_adiabatic_pulse(
    pulse_type: str,
    adiabaticity: int = 4,
    bandwidth: int = 40000,
    beta: int = 800,
    delay: float = 0,
    duration: float = 10e-3,
    dwell: float = 0,
    freq_offset: float = 0,
    max_grad: float = 0,
    max_slew: float = 0,
    n_fac: int = 40,
    mu: float = 4.9,
    phase_offset: float = 0,
    return_gz: bool = False,
    return_delay: bool = False,
    slice_thickness: float = 0,
    system=Opts(),
    use: str = str(),
) -> Union[
    SimpleNamespace,
    Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace],
    Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace],
]:
    """
    Make an adiabatic inversion pulse.

    Note: some parameters only affect certain pulse types and are ignored for other; e.g. bandwidth is ignored if
    type='hypsec'.

    hypsec(n=512, beta=800, mu=4.9, dur=0.012)
        Design a hyperbolic secant adiabatic pulse. `mu` * `beta` becomes the amplitude of the frequency sweep.

        Args:
            - n (int): number of samples (should be a multiple of 4).
            - beta (float): AM waveform parameter.
            - mu (float): a constant, determines amplitude of frequency sweep.
            - dur (float): pulse time (s).

        Returns:
            2-element tuple containing
            - **a** (*array*): AM waveform.
            - **om** (*array*): FM waveform (radians/s).

        References:
            Baum, J., Tycko, R. and Pines, A. (1985). 'Broadband and adiabatic
            inversion of a two-level system by phase-modulated pulses'.
            Phys. Rev. A., 32:3435-3447.

    wurst(n=512, n_fac=40, bw=40000.0, dur=0.002)
        Design a WURST (wideband, uniform rate, smooth truncation) adiabatic inversion pulse

        Args:
            - n (int): number of samples (should be a multiple of 4).
            - n_fac (int): power to exponentiate to within AM term. ~20 or greater is typical.
            - bw (float): pulse bandwidth.
            - dur (float): pulse time (s).

        Returns:
            2-element tuple containing
            - **a** (*array*): AM waveform.
            - **om** (*array*): FM waveform (radians/s).

        References:
            Kupce, E. and Freeman, R. (1995). 'Stretched Adiabatic Pulses for
            Broadband Spin Inversion'.
            J. Magn. Reson. Ser. A., 117:246-256.

    Parameters
    ----------
    pulse_type : str
        One of 'hypsec' or 'wurst' pulse types.
    adiabaticity : int, default=4
    bandwidth : int, default=40000
        Pulse bandwidth.
    beta : int, default=800
        AM waveform parameter.
    delay : float, default=0
        Delay in seconds (s).
    duration : float, default=10e-3
        Pulse time (s).
    dwell : float, default=0
    freq_offset : float, default=0
    max_grad : float, default=0
        Maximum gradient strength.
    max_slew : float, default=0
        Maximum slew rate.
    mu : float, default=4.9
        Constant determining amplitude of frequency sweep.
    n_fac : int, default=40
        Power to exponentiate to within AM term. ~20 or greater is typical.
    phase_offset : float, default=0
        Phase offset.
    return_delay : bool, default=False
        Boolean flag to indicate if the delay has to be returned.
    return_gz : bool, default=False
        Boolean flag to indicate if the slice-selective gradient has to be returned.
    slice_thickness : float, default=0
    system : Opts, default=Opts()
        System limits.
    use : str
        Whether it is a 'refocusing' pulse (for k-space calculation).

    Returns
    -------
    rf : SimpleNamespace
        Adiabatic RF pulse event.
    gz : SimpleNamespace, optional
        Slice-selective trapezoid event.
    gzr : SimpleNamespace, optional
        Slice-select rephasing trapezoid event.
    delay : SimpleNamespace, optional
        Delay event.

    Raises
    ------
    ValueError
        If invalid pulse type is encountered.
        If invalid pulse use is encountered.
        If slice thickness is not provided but slice-selective trapezoid event is expected.
    """
    valid_pulse_types = ["hypsec", "wurst"]
    if pulse_type != "" and pulse_type not in valid_pulse_types:
        raise ValueError(
            f"Invalid type parameter. Must be one of {valid_pulse_types}.Passed: {pulse_type}"
        )
    valid_pulse_uses = get_supported_rf_uses()
    if use != "" and use not in valid_pulse_uses:
        raise ValueError(
            f"Invalid use parameter. Must be one of {valid_pulse_uses}. Passed: {use}"
        )

    if dwell == 0:
        dwell = system.rf_raster_time

    n_raw = np.round(duration / dwell + eps)
    # Number of points must be divisible by 4 - requirement of sigpy.mri
    N = np.floor(n_raw / 4) * 4

    if pulse_type == "hypsec":
        am, fm = hypsec(n=N, beta=beta, mu=mu, dur=duration)
    elif pulse_type == "wurst":
        am, fm = wurst(n=N, n_fac=n_fac, bw=bandwidth, dur=duration)
    else:
        raise ValueError("Unsupported adiabatic pulse type.")

    pm = np.cumsum(fm) * dwell

    ifm = np.argmin(np.abs(fm))
    dfm = np.abs(fm)[ifm]

    # Find rate of change of frequency at the center of the pulse
    if dfm == 0:
        pm0 = pm[ifm]
        am0 = am[ifm]
        roc_fm0 = np.abs(fm[ifm + 1] - fm[ifm - 1]) / 2 / dwell
    else:  # We need to bracket the zero-crossing
        if fm[ifm] * fm[ifm + 1] < 0:
            b = 1
        else:
            b = -1

        pm0 = (pm[ifm] * fm[ifm + b] - pm[ifm + b] * fm[ifm]) / (fm[ifm + b] - fm[ifm])
        am0 = (am[ifm] * fm[ifm + b] - am[ifm + b] * fm[ifm]) / (fm[ifm + b] - fm[ifm])
        roc_fm0 = np.abs(fm[ifm] - fm[ifm + b]) / dwell

    pm -= pm0
    a = (roc_fm0 * adiabaticity) ** 0.5 / 2 / np.pi / am0

    signal = a * am * np.exp(1j * pm)

    if N != n_raw:
        n_pad = n_raw - N
        signal = [
            np.zeros(1, n_pad - np.floor(n_pad / 2)),
            signal,
            np.zeros(1, np.floor(n_pad / 2)),
        ]
        N = n_raw

    t = (np.arange(1, N + 1) - 0.5) * dwell

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
    else:
        rf.use = "inversion"
    if rf.dead_time > rf.delay:
        rf.delay = rf.dead_time

    if return_gz:
        if slice_thickness <= 0:
            raise ValueError("Slice thickness must be provided")

        if max_grad > 0:
            system.max_grad = max_grad

        if max_slew > 0:
            system.max_slew = max_slew

        if pulse_type == "hypsec":
            bandwidth = mu * beta / np.pi
        elif pulse_type == "wurst":
            bandwidth = bandwidth
        else:
            raise ValueError("Unsupported adiabatic pulse type.")

        center_pos, _ = calc_rf_center(rf)

        amplitude = bandwidth / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(
            channel="z", system=system, flat_time=duration, flat_area=area
        )
        gzr = make_trapezoid(
            channel="z",
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz.area - area),
        )

        if rf.delay > gz.rise_time:  # Round-up to gradient raster
            gz.delay = (
                np.ceil((rf.delay - gz.rise_time) / system.grad_raster_time)
                * system.grad_raster_time
            )

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

    if rf.ringdown_time > 0 and return_delay:
        delay = make_delay(calc_duration(rf) + rf.ringdown_time)

    if return_gz and return_delay:
        return rf, gz, gzr, delay
    elif return_gz:
        return rf, gz, gzr
    else:
        return rf
