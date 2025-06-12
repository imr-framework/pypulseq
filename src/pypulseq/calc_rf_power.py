from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np


def calc_rf_power(
    rf: SimpleNamespace,
    dt: float = 1e-6,
    return_peak_power: bool = False,
    return_rf_rms: bool = False,
) -> Union[float, Tuple[float, float], Tuple[float, float, float]]:
    """
    Calculate the relative power of an RF pulse.

    Parameters
    ----------
    rf : SimpleNamespace
        RF pulse event.
    dt : float, default=1e-6
        Time resolution for resampling the RF pulse in (s).
    return_peak_power : bool, default=False
        Boolean flag to indicate if peak power of RF pulse in (Hz**2 s) will be returned.
    return_rf_rms : bool, default=False
        Boolean flag to indicate if RMS of RF pulse in (Hz) will be returned.

    Returns
    -------
    total_energy : float
        Total energy of the RF pulse (Hz^2·s).

    Notes
    -----
    - The RF amplitude is in units of Hz, and the power is in Hz**2.
    - To convert amplitude to Tesla, divide by the gyromagnetic ratio (gamma).
    - To convert power to mT**2·s, divide by gamma**2.
    - The absolute SAR depends on coil and subject-specific factors.

    """
    nn = int(np.round(rf.shape_dur / dt))
    t = (np.arange(nn) + 0.5) * dt
    rfs = np.interp(t, rf.t, rf.signal.real, left=0, right=0) + 1j * np.interp(t, rf.t, rf.signal.imag, left=0, right=0)

    rfs_sq = np.abs(rfs) ** 2
    total_energy = np.sum(rfs_sq) * dt
    peak_pwr = np.max(rfs_sq)
    rf_rms = np.sqrt(total_energy / rf.shape_dur)

    if return_peak_power and not return_rf_rms:
        return total_energy, peak_pwr
    elif return_rf_rms and not return_peak_power:
        return total_energy, rf_rms
    elif return_peak_power and return_rf_rms:
        return total_energy, peak_pwr, rf_rms
    return total_energy
