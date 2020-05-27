from types import SimpleNamespace

import numpy as np


def make_delay(d: float) -> SimpleNamespace:
    """
    Creates a delay event.

    Parameters
    ----------
    d : float
        Delay time in milliseconds (ms).

    Returns
    -------
    delay : SimpleNamespace
        Delay event.
    """

    delay = SimpleNamespace()
    d = d - np.mod(d, 1e-6) # causes crashes on scanners if the precision is better than 1us
    if d < 0:
        raise ValueError('Delay {:.2f} ms is invalid'.format(d * 1e3))
    delay.type = 'delay'
    delay.delay = d
    return delay
