from types import SimpleNamespace
from typing import Union

from pypulseq.block_to_events import block_to_events


def calc_duration(*args: Union[SimpleNamespace, dict, float, int, None]) -> float:
    """
    Calculate the duration of one or more events (or an event block).

    The duration of a single event is defined as the time taken by the event itself plus its delay.
    If multiple events are provided, the returned duration is the maximum of the individual event
    durations (i.e. the duration of a block comprised of those events).

    `None` inputs are ignored, which allows optional events to be passed without an explicit
    conditional. If `None` is the only provided input, a `ValueError` is raised.

    Parameters
    ----------
    args : Union[SimpleNamespace, dict, float, int, None]
        One or more events, a block (i.e. an object with event attributes such as `rf`), label
        events represented as a `dict`, an explicit block duration (`float`/`int`), or `None`.

    Returns
    -------
    duration : float
        Maximum duration of `args`.

    Raises
    ------
    ValueError
        If no non-None inputs are provided.
    """
    if args:
        args_filtered = tuple(a for a in args if a is not None)
        if not args_filtered:
            raise ValueError('At least one non-None input must be provided')
        args = args_filtered

    events = block_to_events(*args)

    duration = 0
    for event in events:
        if event is None:
            continue
        if isinstance(event, (float, int)):  # block_duration field
            assert duration <= event
            duration = event
            continue

        if not isinstance(event, (dict, SimpleNamespace)):
            raise TypeError('input(s) should be of type SimpleNamespace or a dict() in case of LABELINC or LABELSET')

        if event.type == 'delay':
            duration = max(duration, event.delay)
        elif event.type == 'rf':
            duration = max(duration, event.delay + event.shape_dur + event.ringdown_time)
        elif event.type == 'grad':
            duration = max(duration, event.delay + event.shape_dur)
        elif event.type == 'adc':
            duration = max(
                duration,
                event.delay + event.num_samples * event.dwell + event.dead_time,
            )
        elif event.type == 'trap':
            duration = max(
                duration,
                event.delay + event.rise_time + event.flat_time + event.fall_time,
            )
        elif event.type == 'output' or event.type == 'trigger':
            duration = max(duration, event.delay + event.duration)

    return duration
