import numpy as np


def calc_rf_center(rf):
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
