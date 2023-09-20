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
    if len(args) == 1 and hasattr(args[0], 'rf'):
        events = list(vars(args[0]).values())  # Get all attrs
        events = list(
            filter(lambda filter_none: filter_none is not None, events)
        )  # Filter None attributes
        events = __get_label_events_if_any(
            *events
        )  # Flatten label events from dict datatype
        
    else:  # args is a tuple of events
        return args

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
