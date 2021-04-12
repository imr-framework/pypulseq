from types import SimpleNamespace
from typing import Tuple


def block_to_events(args: Tuple[SimpleNamespace, ...]) -> Tuple[SimpleNamespace, ...]:
    """
    Converts `args` from a block to a list of events. If `args` is already a list of event(s), returns it unmodified.

    Parameters
    ----------
    args : list[SimpleNamespace]
        Block to be converted into a list of events, or list of events.

    Returns
    -------
    events : list[SimpleNamespace]
        List of events comprising `args` if it was a block, otherwise `args` unmodified.
    """
    if len(args) == 1:  # args is a tuple consisting either a block or a single event
        x = args[0]
        attrs = vars(x).keys()
        children = [getattr(x, a) for a in attrs]
        if all([isinstance(c, (SimpleNamespace, dict)) for c in children]):  # args is a block of events
            events = list(x.__dict__.values())

            # Are any of the events labels? If yes, extract them from dict()
            for e in events:
                if isinstance(e, dict):
                    events.remove(e)
                    events.extend(e.values())
        else:  # args is a single event
            events = [x]
    else:  # args is a tuple of events
        events = args

    return events
