from types import SimpleNamespace
from typing import Union, Tuple

import numpy as np

from pypulseq.calc_rf_center import calc_rf_center


def calc_rf_bandwidth(
    rf: SimpleNamespace,
    cutoff: float = 0.5,
    df: float = 10.,
    dt: float = 1e-6,
    return_axis: bool = False,
    return_spectrum: bool = False,
) -> Union[float, Tuple[float, np.ndarray], Tuple[float, np.ndarray, float]]:
    """
    Calculate the spectrum of the RF pulse. Returns the bandwidth of the pulse (calculated by a simple FFT, e.g.
    presuming a low-angle approximation) and optionally the spectrum and the frequency axis. The default for the
    optional parameter 'cutoff' is 0.5.
    The default for the optional parameter 'dw' is 10 Hz. The default for 
    the optional parameter 'dt' is 1 us. 

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
    time_center, _ = calc_rf_center(rf)

    # Resample the pulse to a reasonable time array
    nn = np.round(1 / df / dt)
    t = np.arange(-np.floor(nn / 2), np.ceil(nn / 2) - 1) * dt

    rfs = np.interp(xp=rf.t - time_center, fp=rf.signal*np.exp(1j*(rf.phaseOffset+2*np.pi*rf.freqOffset*rf.t)), x=t)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rfs)))
    f = np.arange(-np.floor(nn / 2), np.ceil(nn / 2) - 1) * df

    w1 = __find_flank(f, spectrum, cutoff)
    w2 = __find_flank(f[::-1], spectrum[::-1], cutoff)

    bw = w2 - w1
    fc = (w2+w1)/2

    #  coarse STE scaling -- we normalize to the max of the spectrum, this works
    #  better with frequency-shifted pulses than the abs(sum(shape)) -- the
    #  0-frequency response; yes, we could take the spectrum at fc but this
    #  would have a problem for non-symmetric pulses...
    # s_ref=max(abs(spectrum));
    s_ref = np.interp(f,np.abs(spectrum),fc)
    spectrum = np.sin(2*np.pi*dt*s_ref)*spectrum/s_ref

    if return_spectrum and not return_axis:
        return bw, fc, spectrum, rfs, t
    if return_axis:
        return bw, fc, spectrum, f, rfs, t

    return bw


def __find_flank(x, f, c):
    m = np.max(np.abs(f))
    f = np.abs(f) / m
    i = np.argwhere(f > c)[0]

    return x[i]
