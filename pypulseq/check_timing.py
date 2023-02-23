from types import SimpleNamespace
from typing import Tuple

import numpy as np

from pypulseq import eps
from pypulseq.calc_duration import calc_duration
from pypulseq.opts import Opts


def check_timing(system: Opts, *events: SimpleNamespace) -> Tuple[bool, str, float]:
    """
    Checks if timings of `events` are aligned with the corresponding raster time.

    Parameters
    ----------
    system : Opts
        System limits object.
    events : SimpleNamespace
        Events.

    Returns
    -------
    is_ok : bool
        Boolean flag indicating if timing of events `events` are aligned with gradient raster time
        `system.grad_raster_time`.
    text_err : str
        Error string, if timings are not aligned.
    total_duration : float
        Total duration of events.

    Raises
    ------
    ValueError
        If incorrect data type is encountered in `events`.
    """
    if len(events) == 0:
        text_err = "Empty or damaged block detected"
        is_ok = False
        total_duration = 0.0
        return is_ok, text_err, total_duration

    total_duration = calc_duration(*events)
    is_ok = __div_check(total_duration, system.block_duration_raster)
    text_err = "" if is_ok else f"Total duration: {total_duration * 1e6} us"

    for e in events:
        if isinstance(e, (float, int)):  # Special handling for block_duration
            continue
        elif not isinstance(e, (dict, SimpleNamespace)):
            raise ValueError(
                "Wrong data type of variable arguments, list[SimpleNamespace] expected."
            )
        ok = True
        if isinstance(e, list) and len(e) > 1:
            # For now this is only the case for arrays of extensions, but we cannot actually check extensions anyway...
            continue
        if hasattr(e, "type") and (e.type == "adc" or e.type == "rf"):
            raster = system.rf_raster_time
        else:
            raster = system.grad_raster_time

        if hasattr(e, "delay"):
            if e.delay < -eps:
                ok = False
            if not __div_check(e.delay, raster):
                ok = False

        if hasattr(e, "duration"):
            if not __div_check(e.duration, raster):
                ok = False

        if hasattr(e, "dwell"):
            if (
                e.dwell < system.adc_raster_time
                or np.abs(
                    np.round(e.dwell / system.adc_raster_time) * system.adc_raster_time
                    - e.dwell
                )
                > 1e-10
            ):
                ok = False

        if hasattr(e, "type") and e.type == "trap":
            if (
                not __div_check(e.rise_time, system.grad_raster_time)
                or not __div_check(e.flat_time, system.grad_raster_time)
                or not __div_check(e.fall_time, system.grad_raster_time)
            ):
                ok = False

        if not ok:
            is_ok = False

            text_err = "["
            if hasattr(e, "type"):
                text_err += f"type: {e.type} "
            if hasattr(e, "delay"):
                text_err += f"delay: {e.delay * 1e6} us "
            if hasattr(e, "duration"):
                text_err += f"duration: {e.duration * 1e6} us"
            if hasattr(e, "dwell"):
                text_err += f"dwell: {e.dwell * 1e9} ns"
            if hasattr(e, "type") and e.type == "trap":
                text_err += (
                    f"rise time: {e.rise_time * 1e6} flat time: {e.flat_time * 1e6} "
                    f"fall time: {e.fall_time * 1e6} us"
                )
            text_err += "]"

    return is_ok, text_err, total_duration


def __div_check(a: float, b: float) -> bool:
    """
    Checks whether `a` can be divided by `b` to an accuracy of 1e-9.
    """
    c = a / b
    return abs(c - np.round(c)) < 1e-9
