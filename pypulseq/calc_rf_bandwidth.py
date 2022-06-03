from types import SimpleNamespace
from typing import Union, Tuple

import numpy as np

from pypulseq import calc_rf_center


def calc_rf_bandwidth(
    rf: SimpleNamespace,
    cutoff: float = 0.5,
    return_axis: bool = False,
    return_spectrum: bool = False,
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
    Returns
    -------
    bw : float
        Bandwidth of the RF pulse.

    """
    # TODO CHECK
    time_center, _ = calc_rf_center(rf)

    # Resample the pulse to a reasonable time array
    dw = 10  # Hz
    dt = 1e-6  # For now, 1 Mhz
    nn = np.round(1 / dw / dt)
    tt = np.arange(-np.floor(nn / 2), np.ceil(nn / 2) - 1) * dt

    rfs = np.interp(xp=rf.t - time_center, fp=rf.sig, x=tt)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rfs)))
    w = np.arange(-np.floor(nn / 2), np.ceil(nn / 2) - 1) * dw

    w1 = __find_flank(w, spectrum, cutoff)
    w2 = __find_flank(w[::-1], spectrum[::-1], cutoff)

    bw = w2 - w1

    if return_spectrum and not return_axis:
        return bw, spectrum
    if return_axis:
        return bw, spectrum, w

    return bw


def __find_flank(x, f, c):
    m = np.max(np.abs(f))
    f = np.abs(f) / m - c
    i = (f > 0)[0, 0]

    if i > 1:
        f0 = f[i - 1]
        f1 = f[i]
        xf = (f1 * x[i - 1] - f0 * x[i]) / (f1 - f0)
    else:
        xf = x[0]

    return xf
