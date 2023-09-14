from types import SimpleNamespace

import numpy as np

from pypulseq.block_to_events import block_to_events


def calc_duration(*args: SimpleNamespace) -> float:
    """
    Calculate the duration of an event or block.

    Parameters
    ----------
    args : SimpleNamespace
        Block or events.

    Returns
    -------
    duration : float
        Cumulative duration of `args`.
    """
    events = block_to_events(*args)

    duration = 0
    for event in events:
        if isinstance(event, (float, int)):  # block_duration field
            assert duration <= event
            duration = event
            continue

        if not isinstance(event, (dict, SimpleNamespace)):
            raise TypeError(
                "input(s) should be of type SimpleNamespace or a dict() in case of LABELINC or LABELSET"
            )

        if event.type == "delay":
            duration = max(duration, event.delay)
        elif event.type == "rf":
            duration = max(
                duration, event.delay + event.shape_dur + event.ringdown_time
            )
        elif event.type == "grad":
            duration = max(duration, event.delay + event.shape_dur)
        elif event.type == "adc":
            duration = max(
                    duration,
                    event.delay + event.num_samples * event.dwell + event.dead_time,
            )
        elif event.type == "trap":
            duration = max(
                    duration,
                    event.delay + event.rise_time + event.flat_time + event.fall_time,
            )
        elif event.type == "output" or event.type == "trigger":
            duration = max(duration, event.delay + event.duration)

    return duration
