from types import SimpleNamespace
from typing import Tuple

import numpy as np


def calc_rf_center(rf: SimpleNamespace) -> Tuple[float, float]:
    """
    Calculate the time point of the effective rotation calculated as the peak of the radio-frequency amplitude for the
    shaped pulses and the center of the pulse for the block pulses. Zero padding in the radio-frequency pulse is
    considered as a part of the shape. Delay field of the radio-frequency object is not taken into account.

    Parameters
    ----------
    rf : SimpleNamespace
        Radio-frequency pulse event.

    Returns
    -------
    time_center : float
        Time point of the center of the radio-frequency pulse.
    id_center : float
        Corresponding position of `time_center` in the radio-frequency pulse's envelope.
    """
    # Detect the excitation peak; if i is a plateau take its center
    rf_max = np.max(np.abs(rf.signal))
    i_peak = np.where(np.abs(rf.signal) >= rf_max * 0.99999)[0]
    time_center = (rf.t[i_peak[0]] + rf.t[i_peak[-1]]) / 2
    id_center = i_peak[round((len(i_peak) - 1) / 2)]

    return time_center, id_center
