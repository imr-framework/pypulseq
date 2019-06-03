from types import SimpleNamespace


def make_delay(d):
    """
    Makes a Holder object for an delay Event.

    Parameters
    ----------
    d : float
        Delay time, in seconds.

    Returns
    -------
    delay : Holder
        Delay Event.
    """

    delay = SimpleNamespace()
    if d < 0:
        raise ValueError('Delay {:.2f} ms is invalid'.format(d * 1e3))
    delay.type = 'delay'
    delay.delay = d
    return delay
