from types import SimpleNamespace
from typing import Tuple


def block_to_events(*args: SimpleNamespace) -> Tuple[SimpleNamespace, ...]:
    """
    Converts `args` from a block to a list of events. If `args` is already a list of event(s), returns it unmodified.

    Parameters
    ----------
    args : SimpleNamespace
        Block to be flattened into a list of events.

    Returns
    -------
    events : list[SimpleNamespace]
        List of events comprising `args` if it was a block, otherwise `args` unmodified.
    """
    if (
        len(args) == 1
    ):  # args is a tuple consisting a block of events, or a single event
        x = args[0]

        if isinstance(x, (float, int)):  # args is block duration
            events = [x]
        else:  # args could be a block of events or a single event
            events = list(vars(x).values())  # Get all attrs
            events = list(
                filter(lambda filter_none: filter_none is not None, events)
            )  # Filter None attributes
            # If all attrs are either float/SimpleNamespace, args is a block of events
            if all([isinstance(e, (float, SimpleNamespace)) for e in events]):
                events = __get_label_events_if_any(
                    *events
                )  # Flatten label events from dict datatype
            else:  # Else, args is a single event
                events = [x]
    else:  # args is a tuple of events
        events = __get_label_events_if_any(*args)

    return events


def __get_label_events_if_any(*events: list) -> list:
    # Are any of the events labels? If yes, extract them from dict()
    final_events = []
    for e in events:
        if isinstance(e, dict):  # Only labels are stored as dicts
            final_events.extend(e.values())
        else:
            final_events.append(e)

    return final_events
