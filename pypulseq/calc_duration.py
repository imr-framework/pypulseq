from types import SimpleNamespace


def calc_duration(*events: list) -> float:
    """
    Calculate the cumulative duration of Events.

    Parameters
    ----------
    events : list
        List of `SimpleNamespace` events. Can also be a list containing a single block (see
        `pypulseq.Sequence.sequence.plot()`).

    Returns
    -------
    duration : float
        The cumulative duration of the pulse events in `events`.
    """

    for e in events:
        if isinstance(e, SimpleNamespace) and isinstance(getattr(e, list(e.__dict__.keys())[0]), SimpleNamespace):
            events = list(e.__dict__.values())
            break

    duration = 0
    for event in events:
        if not isinstance(event, SimpleNamespace):
            raise TypeError("input(s) should be of type SimpleNamespace")

        if event.type == 'delay':
            duration = max(duration, event.delay)
        elif event.type == 'rf':
            duration = max(duration, event.delay + event.t[-1])
        elif event.type == 'grad':
            duration = max(duration, event.t[-1] + event.t[1] - event.t[0] + event.delay)
        elif event.type == 'adc':
            duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)
        elif event.type == 'trap':
            duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)

    return duration
