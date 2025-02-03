import math
from types import SimpleNamespace
from typing import Tuple, Union
from warnings import warn

import numpy as np

from pypulseq import eps
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses
from pypulseq.utils.tracing import trace, trace_enabled


def make_adiabatic_pulse(
    pulse_type: str,
    adiabaticity: int = 4,
    bandwidth: int = 40000,
    beta: float = 800.0,
    delay: float = 0,
    duration: float = 10e-3,
    dwell: Union[float, None] = None,
    freq_offset: float = 0,
    max_grad: Union[float, None] = None,
    max_slew: Union[float, None] = None,
    n_fac: int = 40,
    mu: float = 4.9,
    phase_offset: float = 0,
    return_gz: bool = False,
    slice_thickness: float = 0,
    system: Union[Opts, None] = None,
    use: str = str(),
) -> Union[
    SimpleNamespace,
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

    Returns
    -------
            2-element tuple containing
            - **a** (*array*): AM waveform.
            - **om** (*array*): FM waveform (radians/s).

    References
    ----------
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

    Returns
    -------
            2-element tuple containing
            - **a** (*array*): AM waveform.
            - **om** (*array*): FM waveform (radians/s).

    References
    ----------
            Kupce, E. and Freeman, R. (1995). 'Stretched Adiabatic Pulses for
            Broadband Spin Inversion'.
            J. Magn. Reason. Set. A., 117:246-256.

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
    dwell : float, default=None
    freq_offset : float, default=0
    max_grad : float, default=None
        Maximum gradient strength.
    max_slew : float, default=None
        Maximum slew rate.
    mu : float, default=4.9
        Constant determining amplitude of frequency sweep.
    n_fac : int, default=40
        Power to exponentiate to within AM term. ~20 or greater is typical.
    phase_offset : float, default=0
        Phase offset.
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

    Raises
    ------
    ValueError
        If invalid pulse type is encountered.
        If invalid pulse use is encountered.
        If slice thickness is not provided but slice-selective trapezoid event is expected.
    """
    if system is None:
        system = Opts.default

    if return_gz and slice_thickness <= 0:
        raise ValueError('Slice thickness must be provided')

    valid_pulse_types = ['hypsec', 'wurst']
    if (not pulse_type) or (pulse_type not in valid_pulse_types):
        raise ValueError(f'Invalid type parameter. Must be one of {valid_pulse_types}.Passed: {pulse_type}')
    valid_rf_use_labels = get_supported_rf_uses()
    if use != '' and use not in valid_rf_use_labels:
        raise ValueError(f'Invalid use parameter. Must be one of {valid_rf_use_labels}. Passed: {use}')

    if dwell is None:
        dwell = system.rf_raster_time

    # WTC and PS - we have no idea why eps is added here. Leaving for now.
    n_raw = round(duration / dwell + eps)
    # Number of points must be divisible by 4 - requirement of individual pulse functions
    n_samples = math.floor(n_raw / 4) * 4

    if pulse_type == 'hypsec':
        amp_mod, freq_mod = _hypsec(n=n_samples, beta=beta, mu=mu, dur=duration)
    elif pulse_type == 'wurst':
        amp_mod, freq_mod = _wurst(n=n_samples, n_fac=n_fac, bw=bandwidth, dur=duration)

    phase_mod = np.cumsum(freq_mod) * dwell

    min_abs_freq_idx = np.argmin(np.abs(freq_mod))
    min_abs_freq_value = abs(freq_mod[min_abs_freq_idx])

    # Find rate of change of frequency at the center of the pulse
    if min_abs_freq_value == 0:
        phase_at_zero_freq = phase_mod[min_abs_freq_idx]
        amp_at_zero_freq = amp_mod[min_abs_freq_idx]
        rate_of_freq_change = abs(freq_mod[min_abs_freq_idx + 1] - freq_mod[min_abs_freq_idx - 1]) / (2 * dwell)
    else:  # We need to bracket the zero-crossing
        b = 1 if freq_mod[min_abs_freq_idx] * freq_mod[min_abs_freq_idx + 1] < 0 else -1
        diff_freq = freq_mod[min_abs_freq_idx + b] - freq_mod[min_abs_freq_idx]

        phase_at_zero_freq = (
            phase_mod[min_abs_freq_idx] * freq_mod[min_abs_freq_idx + b]
            - phase_mod[min_abs_freq_idx + b] * freq_mod[min_abs_freq_idx]
        ) / diff_freq

        amp_at_zero_freq = (
            amp_mod[min_abs_freq_idx] * freq_mod[min_abs_freq_idx + b]
            - amp_mod[min_abs_freq_idx + b] * freq_mod[min_abs_freq_idx]
        ) / diff_freq

        rate_of_freq_change = abs(freq_mod[min_abs_freq_idx] - freq_mod[min_abs_freq_idx + b]) / dwell

    # Adjust phase modulation and calculate amplitude
    phase_mod -= phase_at_zero_freq
    amp = np.sqrt(rate_of_freq_change * adiabaticity) / (2 * np.pi * amp_at_zero_freq)

    # Create the modulated signal
    signal = amp * amp_mod * np.exp(1j * phase_mod)

    # Adjust the number of samples if needed
    if n_samples != n_raw:
        n_pad = n_raw - n_samples
        pad_left = n_pad // 2
        pad_right = n_pad - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
        n_samples = n_raw

    # Calculate time points
    t = (np.arange(n_samples) + 0.5) * dwell

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.shape_dur = n_samples * dwell
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay
    rf.use = use if use != '' else 'inversion'
    if rf.dead_time > rf.delay:
        warn(
            f'Specified RF delay {rf.delay * 1e6:.2f} us is less than the dead time {rf.dead_time * 1e6:.0f} us. Delay was increased to the dead time.',
            stacklevel=2,
        )
        rf.delay = rf.dead_time

    if return_gz:
        max_grad_slice_select = max_grad
        max_slew_slice_select = max_slew

        if pulse_type == 'hypsec':
            bandwidth = mu * beta / np.pi
        elif pulse_type == 'wurst':
            bandwidth = bandwidth

        center_pos, _ = calc_rf_center(rf)

        amplitude = bandwidth / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(
            channel='z',
            system=system,
            flat_time=duration,
            flat_area=area,
            max_grad=max_grad_slice_select,
            max_slew=max_slew_slice_select,
        )
        gzr = make_trapezoid(
            channel='z',
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz.area - area),
            max_grad=max_grad_slice_select,
            max_slew=max_slew_slice_select,
        )

        if rf.delay > gz.rise_time:  # Round-up to gradient raster
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

    if trace_enabled():
        rf.trace = trace()

    if return_gz:
        return rf, gz, gzr
    else:
        return rf


"""Adiabatic Pulse Design functions.
    The below functions are originally from ssigpy/sigpy/mri/rf/adiabatic.py
    Used under the terms of the Sigpy BSD 3-clause license.

    Copyright (c) 2016, Frank Ong.
    Copyright (c) 2016, The Regents of the University of California.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in
       the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
    OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def _bir4(n: int, beta: float, kappa: float, theta: float, dw0: np.ndarray):
    r"""Design a BIR-4 adiabatic pulse.

    BIR-4 is equivalent to two BIR-1 pulses back-to-back.

    Args:
        n (int): number of samples (should be a multiple of 4).
        beta (float): AM waveform parameter.
        kappa (float): FM waveform parameter.
        theta (float): flip angle in radians.
        dw0: FM waveform scaling (radians/s).

    Returns
    -------
        2-element tuple containing

        - **a** (*array*): AM waveform.
        - **om** (*array*): FM waveform (radians/s).

    References
    ----------
        Staewen, R.S. et al. (1990). '3-D FLASH Imaging using a single surface
        coil and a new adiabatic pulse, BIR-4'.
        Invest. Radiology, 25:559-567.
    """
    dphi = np.pi + theta / 2

    t = np.arange(0, n) / n

    a1 = np.tanh(beta * (1 - 4 * t[: n // 4]))
    a2 = np.tanh(beta * (4 * t[n // 4 : n // 2] - 1))
    a3 = np.tanh(beta * (3 - 4 * t[n // 2 : 3 * n // 4]))
    a4 = np.tanh(beta * (4 * t[3 * n // 4 :] - 3))

    a = np.concatenate((a1, a2, a3, a4)).astype(np.complex64)
    a[n // 4 : 3 * n // 4] = a[n // 4 : 3 * n // 4] * np.exp(1j * dphi)

    om1 = dw0 * np.tan(kappa * 4 * t[: n // 4]) / np.tan(kappa)
    om2 = dw0 * np.tan(kappa * (4 * t[n // 4 : n // 2] - 2)) / np.tan(kappa)
    om3 = dw0 * np.tan(kappa * (4 * t[n // 2 : 3 * n // 4] - 2)) / np.tan(kappa)
    om4 = dw0 * np.tan(kappa * (4 * t[3 * n // 4 :] - 4)) / np.tan(kappa)

    om = np.concatenate((om1, om2, om3, om4))

    return a, om


def _hypsec(n: int = 512, beta: float = 800.0, mu: float = 4.9, dur: float = 0.012):
    r"""Design a hyperbolic secant adiabatic pulse.

    mu * beta becomes the amplitude of the frequency sweep

    Args:
        n (int): number of samples (should be a multiple of 4).
        beta (float): AM waveform parameter.
        mu (float): a constant, determines amplitude of frequency sweep.
        dur (float): pulse time (s).

    Returns
    -------
        2-element tuple containing

        - **a** (*array*): AM waveform.
        - **om** (*array*): FM waveform (radians/s).

    References
    ----------
        Baum, J., Tycko, R. and Pines, A. (1985). 'Broadband and adiabatic
        inversion of a two-level system by phase-modulated pulses'.
        Phys. Rev. A., 32:3435-3447.
    """
    t = np.arange(-n // 2, n // 2) / n * dur

    a = np.cosh(beta * t) ** -1
    om = -mu * beta * np.tanh(beta * t)

    return a, om


def _wurst(n: int = 512, n_fac: int = 40, bw: float = 40e3, dur: float = 2e-3):
    r"""Design a WURST (wideband, uniform rate, smooth truncation) adiabatic
     inversion pulse

    Args:
        n (int): number of samples (should be a multiple of 4).
        n_fac (int): power to exponentiate to within AM term. ~20 or greater is
         typical.
        bw (float): pulse bandwidth.
        dur (float): pulse time (s).


    Returns
    -------
        2-element tuple containing

        - **a** (*array*): AM waveform.
        - **om** (*array*): FM waveform (radians/s).

    References
    ----------
        Kupce, E. and Freeman, R. (1995). 'Stretched Adiabatic Pulses for
        Broadband Spin Inversion'.
        J. Magn. Reason. Set. A., 117:246-256.
    """
    t = np.arange(0, n) * dur / n

    a = 1 - np.power(np.abs(np.cos(np.pi * t / dur)), n_fac)
    om = np.linspace(-bw / 2, bw / 2, n) * 2 * np.pi

    return a, om


def _goia_wurst(
    n: int = 512,
    dur: float = 3.5e-3,
    f: float = 0.9,
    n_b1: int = 16,
    m_grad: int = 4,
    b1_max: float = 817,
    bw: float = 20000,
):
    r"""Design a GOIA (gradient offset independent adiabaticity) WURST
     inversion pulse

    Args:
        n (int): number of samples.
        dur (float): pulse duration (s).
        f (float): [0,1] gradient modulation factor
        n_b1 (int): order for B1 modulation
        m_grad (int): order for gradient modulation
        b1_max (float): maximum b1 (Hz)
        bw (float): pulse bandwidth (Hz)

    Returns
    -------
        3-element tuple containing:

        - **a** (*array*): AM waveform (Hz)
        - **om** (*array*): FM waveform (Hz)
        - **g** (*array*): normalized gradient waveform

    References
    ----------
        O. C. Andronesi, S. Ramadan, E.-M. Ratai, D. Jennings, C. E. Mountford,
        A. G. Sorenson.
        J Magn Reason, 203:283-293, 2010.

    """
    t = np.arange(0, n) * dur / n

    a = b1_max * (1 - np.abs(np.sin(np.pi / 2 * (2 * t / dur - 1))) ** n_b1)
    g = (1 - f) + f * np.abs(np.sin(np.pi / 2 * (2 * t / dur - 1))) ** m_grad
    om = np.cumsum((a**2) / g) * dur / n
    om = om - om[n // 2 + 1]
    om = g * om
    om = om / np.max(np.abs(om)) * bw / 2

    return a, om, g


def _bloch_siegert_fm(
    n: int = 512,
    dur: float = 2e-3,
    b1p: float = 20.0,
    k: float = 42.0,
    gamma: Union[float, None] = None,
):
    r"""
    U-shaped FM waveform for adiabatic Bloch-Siegert :math:`B_1^{+}` mapping
    and spatial encoding.

    Args:
        n (int): number of time points
        dur (float): duration in seconds
        b1p (float): nominal amplitude of constant AM waveform
        k (float): design parameter that affects max in-band
            perturbation
        gamma (float): gyromagnetic ratio

    Returns
    -------
        om (array): FM waveform (radians/s).

    References
    ----------
        M. M. Khalighi, B. K. Rutt, and A. B. Kerr.
        Adiabatic RF pulse design for Bloch-Siegert B1+ mapping.
        Magn Reason Med, 70(3):829-835, 2013.

        M. Jankiewicz, J. C. Gore, and W. A. Grissom.
        Improved encoding pulses for Bloch-Siegert B1+ mapping.
        J Magn Reason, 226:79-87, 2013.

    """
    # set gamma to PyPulseq default if not provided
    if gamma is None:
        gamma = 2 * np.pi * 42.576e6

    t = np.arange(1, n // 2) * dur / n

    om = gamma * b1p / np.sqrt((1 - gamma * b1p / k * t) ** -2 - 1)
    om = np.concatenate((om, om[::-1]))

    return om
