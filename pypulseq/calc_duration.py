from pypulseq.holder import Holder


def calc_duration(*events):
    """
    Calculate the cumulative duration of Events.

    Parameters
    ----------
    events : list
        List of Holder objects. Can also be a list containing a single block (see plot() in Sequence).

    Returns
    -------
    duration : float
        The cumulative duration of the Events passed.
    """
    if isinstance(events[0], dict):
        events = events[0].values()

    duration = 0
    for event in events:
        if not isinstance(event, Holder):
            raise TypeError("input(s) should be of type core.events.Holder")

        if event.type == 'delay':
            duration = max(duration, event.delay)
        elif event.type == 'rf':
            duration = max(duration, event.t[0][-1] + event.dead_time + event.ring_down_time)
        elif event.type == 'grad':
            duration = max(duration, event.t[0][-1])
        elif event.type == 'adc':
            adc_time = event.delay + event.num_samples * event.dwell + event.dead_time
            duration = max(duration, adc_time)
        elif event.type == 'trap':
            duration = max(duration, event.rise_time + event.flat_time + event.fall_time)

    return duration
