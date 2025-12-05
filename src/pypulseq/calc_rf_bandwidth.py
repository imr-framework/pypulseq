import math
import warnings
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.opts import Opts


def calc_rf_bandwidth(
    rf: SimpleNamespace,
    cutoff: float = 0.5,
    return_axis: bool = False,
    return_spectrum: bool = False,
    dw: float = 10,
    dt: Union[float, None] = None,
) -> Union[float, Tuple[float, np.ndarray], Tuple[float, np.ndarray, float]]:
    """
    Calculate the spectrum of the RF pulse. Returns the bandwidth of the pulse (calculated by a simple FFT, e.g.
    presuming a low-angle approximation) and optionally the spectrum and the frequency axis. The default for the
    optional parameter 'cutoff' is 0.5.

    Parameters
    ----------
    rf : SimpleNamespace
        RF pulse event.
    cutoff : float, default=0.5
    return_axis : bool, default=False
        Boolean flag to indicate if frequency axis of RF pulse will be returned.
    return_spectrum : bool, default=False
        Boolean flag to indicate if spectrum of RF pulse will be returned.
    dw : float, default=10
        Spectral resolution in (Hz).
    dt : Union[float, None], default=None
        Sampling time in (s). Defaults to Opts().rf_raster_time.

    Returns
    -------
    bw : float
        Bandwidth of the RF pulse.

    """
    if dt is None:
        dt = Opts().rf_raster_time

    time_center, _ = calc_rf_center(rf)

    if abs(rf.freq_ppm) > np.finfo(float).eps:
        warnings.warn(
            'calc_rf_bandwidth(): relying on the system properties, like B0 and gamma, '
            'stored in the global environment by calling pypulseq.Opts()'
        )
        sys = Opts()
        full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * sys.gamma * sys.B0
    else:
        full_freq_offset = rf.freq_offset

    # Resample the pulse to a reasonable time array
    nn = round(1 / dw / dt)
    tt = np.arange(-math.floor(nn / 2), math.ceil(nn / 2) - 1) * dt

    rf_signal = rf.signal * np.exp(1j * rf.phase_offset + 2 * math.pi * full_freq_offset * rf.t)
    rfs = np.interp(xp=rf.t - time_center, fp=rf_signal, x=tt)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rfs)))
    w = np.arange(-math.floor(nn / 2), math.ceil(nn / 2) - 1) * dw

    w1 = __find_flank(w, spectrum, cutoff)
    w2 = __find_flank(w[::-1], spectrum[::-1], cutoff)

    bw = w2 - w1

    if return_spectrum and not return_axis:
        return bw, spectrum
    elif return_axis and not return_spectrum:
        return bw, w
    elif return_spectrum and return_axis:
        return bw, spectrum, w

    return bw


def __find_flank(x, f, c):
    m = np.max(np.abs(f))
    f = np.abs(f) / m
    i = np.argwhere(f > c)[0]

    return x[i]
