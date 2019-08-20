import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.opts import Opts
from types import SimpleNamespace


def check_timing(system: Opts, *events: SimpleNamespace):
    """
    Checks if timings of events `events` are aligned with gradient raster time `system.grad_raster_time`.

    Parameters
    ----------
    system : Opts
        System limits object.
    events : SimpleNamespace
        Pulse events.

    Returns
    -------
    is_ok : bool
        Boolean flag indicating if timing of events `events` are aligned with gradient raster time
        `system.grad_raster_time`.
    text_err : str
        Error string, if timings are not aligned.
    """
    total_dur = calc_duration(*events)
    is_ok = __div_check(total_dur, system.grad_raster_time)
    if is_ok:
        text_err = str()
    else:
        text_err = f'Total duration: {total_dur * 1e6} us'

    for i in range(len(events)):
        e = events[i]
        ok = True
        if hasattr(e, 'type') and e.type == 'adc' or e.type == 'rf':
            raster = system.rf_raster_time
        else:
            raster = system.grad_raster_time

        if hasattr(e, 'delay'):
            if not __div_check(e.delay, raster):
                ok = False

        if hasattr(e, 'type') and e.type == 'trap':
            if not __div_check(e.rise_time, system.grad_raster_time) or \
                    not __div_check(e.flat_time, system.grad_raster_time) or \
                    not __div_check(e.fall_time, system.grad_raster_time):
                ok = False

        if not ok:
            is_ok = False
            if len(text_err) != 0:
                text_err += ' '

            text_err += '[ '
            if hasattr(e, 'type'):
                text_err += f'type: {e.type} '
            if hasattr(e, 'delay'):
                text_err += f'delay: {e.delay * 1e6} us '
            if hasattr(e, 'type') and e.type == 'trap':
                text_err += f'rise time: {e.rise_time * 1e6} flat time: {e.flat_time * 1e6} ' \
                            f'fall time: {e.fall_time * 1e6} us '
            text_err += ']'

    return is_ok, text_err


def __div_check(a, b):
    c = a / b
    return abs(c - np.round(c)) < 1e-9
