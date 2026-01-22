from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np


def calc_rf_power(
    rf: SimpleNamespace,
    dt: Union[float, None] = None,
) -> Tuple[float, float, float]:
    """
    Calculate the relative power of the RF pulse.

    Returns the (relative) energy of the pulse expressed in the units of
    RF amplitude squared multiplied by time, e.g. in Pulseq these are
    Hz * Hz * s = Hz. Sounds strange, but is true.

    Parameters
    ----------
    rf : SimpleNamespace
        Pulseq RF Event.
    dt : float, default=None
        RF raster time in s. If provided, resample signal to
        the target uniform raster, otherwise directly use pulse
        't' attribute. The default is None.

    Returns
    -------
    total_energy : float
        Relative power of the RF pulse in Hz.
    peak_pwr : float
        Peak power of the RF pulse in Hz**2.
    rf_rms : float
        RMS B1 amplitude of the RF pulse in Hz.

    Notes
    -----
    The power and rf amplitude calculated by this function is
    relative as it is calculated in units of Hz**2 or Hz. The rf amplitude
    can be converted to T by dividing the resulting value by gamma.
    Correspondingly, The power can be converted to mT**2 * s by dividing
    the given value by gamma^2. Nonetheless, the absolute SAR is related to
    the electric field, so the further scaling coeficient is both tx-coil-
    dependent (e.g. depends on the coil design) and also subject-dependent
    (e.g. depends on the reference voltage).

    """
    t = np.asarray(rf.t)
    s = np.asarray(rf.signal)

    if dt is None:
        # Force interpolation
        rfs_sq = np.abs(s) ** 2
        total_energy = np.trapezoid(rfs_sq, t)
        peak_pwr = np.max(rfs_sq)
        shape_dur = t[-1] - t[0]
        rf_rms = np.sqrt(total_energy / shape_dur)
        return total_energy, peak_pwr, rf_rms

    else:
        # Resample
        nn = round(rf.shape_dur / dt)
        t_new = (np.arange(nn) + 0.5) * dt
        s_new = np.interp(t_new, t, s, left=0, right=0)

        rfs_sq = np.abs(s_new) ** 2
        total_energy = np.sum(rfs_sq) * dt
        peak_pwr = np.max(rfs_sq)
        rf_rms = np.sqrt(total_energy / rf.shape_dur)
        return total_energy, peak_pwr, rf_rms
