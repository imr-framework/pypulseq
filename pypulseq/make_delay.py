from types import SimpleNamespace


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
    if d < 0:
        raise ValueError('Delay {:.2f} ms is invalid'.format(d * 1e3))
    delay.type = 'delay'
    delay.delay = d
    return delay
