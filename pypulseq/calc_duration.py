from types import SimpleNamespace

from pypulseq.block_to_events import block_to_events


def calc_duration(*args: SimpleNamespace) -> float:
    """
    Calculate the cumulative duration of Events.

    Parameters
    ----------
    args : list[SimpleNamespace]
        List of `SimpleNamespace` objects. Can also be a list containing a single block (see
        `pypulseq.Sequence.sequence.plot()`).

    Returns
    -------
    duration : float
        The cumulative duration of the pulse events in `events`.
    """
    events = block_to_events(args)

    duration = 0
    for event in events:
        if not isinstance(event, (dict, SimpleNamespace)):
            raise TypeError("input(s) should be of type SimpleNamespace or a dict() in case of LABELINC or LABELSET")

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
        elif event.type == 'output' or event.type == 'trigger':
            duration = max(duration, event.delay + event.duration)

    return duration
