from __future__ import annotations

import numpy as np


def do_sw_sar(SARwbg_lim_s: np.ndarray, tsec: np.ndarray, t: int) -> np.ndarray:
    """Compute sliding window time-averaged SAR over window length t.

    Equivalent to MATLAB do_sw_sar.
    """

    SAR_timeavg = np.zeros(len(tsec), dtype=float)
    # MATLAB range 1:(tsec(end)-t) corresponds to Python 0..(tsec[-1]-t-1)
    for instant in range(0, int(tsec[-1] - t)):
        SAR_timeavg[instant] = float(np.sum(SARwbg_lim_s[instant : instant + t]) / t)
    return SAR_timeavg



