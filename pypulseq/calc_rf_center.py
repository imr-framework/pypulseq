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
    eps = np.finfo(float).eps
    for first, x in enumerate(rf.signal):
        if abs(x) > eps:
            break

    for last, x in enumerate(rf.signal[::-1]):
        if abs(x) > eps:
            break

    # Detect the excitation peak: we traverse over in-place reverse of rf.signal; we want index from the ending
    last = len(rf.signal) - last - 1
    rf_min = min(abs(rf.signal[first:last + 1]))
    rf_max = max(abs(rf.signal[first:last + 1]))
    id_center = np.argmax(abs(rf.signal[first:last + 1]))
    if rf_max - rf_min <= eps:
        id_center = round((last - first + 1) / 2) - 1

    time_center = rf.t[first + id_center]

    return time_center, id_center
