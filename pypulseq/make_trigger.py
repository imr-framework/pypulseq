# inserted for trigger support by mveldmann

from types import SimpleNamespace
from typing import Union

from pypulseq.opts import Opts


def make_trigger(
    channel: str, delay: float = 0, duration: float = 0, system: Union[Opts, None] = None
) -> SimpleNamespace:
    """
     Create a trigger halt event for a synchronization with an external signal from a given channel with an optional
     given delay prio to the sync and duration after the sync. Possible channel values: 'physio1','physio2'
     (Siemens specific).

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
     channel : str
         Must be one of 'physio1' or 'physio2'.
     delay : float, default=0
         Delay in seconds
     duration: float, default=0
         Duration in seconds.
     system : Opts, default=Opts()
         System limits.

    Returns
    -------
     trigger : SimpleNamespace
         Trigger event.

    Raises
    ------
     ValueError
         If invalid `channel` is passed. Must be one of 'physio1' or 'physio2'.
    """
    if system is None:
        system = Opts.default

    if channel not in ['physio1', 'physio2']:
        raise ValueError(f"Channel {channel} is invalid. Must be one of 'physio1' or 'physio2'.")

    trigger = SimpleNamespace()
    trigger.type = 'trigger'
    trigger.channel = channel
    trigger.delay = delay
    trigger.duration = duration
    if trigger.duration <= system.grad_raster_time:
        trigger.duration = system.grad_raster_time

    return trigger
