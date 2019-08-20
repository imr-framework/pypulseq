from types import SimpleNamespace

import numpy as np


def calc_rf_center(rf: SimpleNamespace):
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
    tc : float
        Time point of the center of the radio-frequency pulse.
    ic : float
        Corresponding position of `tc` in the radio-frequency pulse's envelope.
    """
    eps = np.finfo(float).eps
    for first, x in enumerate(rf.signal):
        if abs(x) > eps:
            break

    for last, x in enumerate(rf.signal[::-1]):
        if abs(x) > eps:
            break

    last = len(rf.signal) - last - 1  # Because we traverse over in-place reverse of rf.signal
    rf_min = min(abs(rf.signal[first:last]))
    rf_max, ic = max(abs(rf.signal[first:last])), np.argmax(abs(rf.signal[first:last]))
    if rf_max - rf_min <= eps:
        ic = round((last - first + 1) / 2)

    tc = rf.t[first - 1 + ic]

    return tc, ic
