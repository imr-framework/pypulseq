from types import SimpleNamespace

import numpy as np

from pypulseq import eps
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform


def make_extended_trapezoid(
    channel: str,
    amplitudes: np.ndarray = np.zeros(1),
    convert_to_arbitrary: bool = False,
    max_grad: float = 0,
    max_slew: float = 0,
    skip_check: bool = False,
    system: Opts = Opts(),
    times: np.ndarray = np.zeros(1),
) -> SimpleNamespace:
    """
    Create a gradient by specifying a set of points (amplitudes) at specified time points(times) at a given channel
    with given system limits. Returns an arbitrary gradient object.

    See also:
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.opts.Opts`
    - `pypulseq.make_trapezoid.make_trapezoid()`

    Parameters
    ----------
    channel : str
        Orientation of extended trapezoidal gradient event. Must be one of 'x', 'y' or 'z'.
    convert_to_arbitrary : bool, default=False
        Boolean flag to indicate if the extended trapezoid gradient has to be converted into an arbitrary gradient.
    amplitudes : numpy.ndarray, default=09
        Values defined at `times` time indices.
    max_grad : float, default=0
        Maximum gradient strength.
    max_slew : float, default=0
        Maximum slew rate.
    system : Opts, default=Opts()
        System limits.
    skip_check : bool, default=False
        Boolean flag to indicate if amplitude check is to be skipped.
    times : numpy.ndarray, default=np.zeros(1)
        Time points at which `amplitudes` defines amplitude values.

    Returns
    -------
    grad : SimpleNamespace
        Extended trapezoid gradient event.

    Raises
    ------
    ValueError
        If invalid `channel` is passed. Must be one of 'x', 'y' or 'z'.
        If all elements in `times` are zero.
        If elements in `times` are not in ascending order or not distinct.
        If all elements in `amplitudes` are zero.
        If first amplitude of a gradient is non-ero and does not connect to a previous block.
    """
    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of 'x', 'y' or 'z'. Passed: {channel}"
        )

    times = np.asarray(times)
    amplitudes = np.asarray(amplitudes)

    if len(times) != len(amplitudes):
        raise ValueError("Times and amplitudes must have the same length.")

    if np.all(times == 0):
        raise ValueError("At least one of the given times must be non-zero")

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "Times must be in ascending order and all times must be distinct"
        )

    if (
        np.abs(
            np.round(times[-1] / system.grad_raster_time) * system.grad_raster_time
            - times[-1]
        )
        > eps
    ):
        raise ValueError("The last time point must be on a gradient raster")

    if skip_check is False and times[0] > 0 and amplitudes[0] != 0:
        raise ValueError(
            "If first amplitude of a gradient is non-zero, it must connect to previous block"
        )

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    if convert_to_arbitrary:
        # Represent the extended trapezoid on the regularly sampled time grid
        waveform = points_to_waveform(
            times=times, amplitudes=amplitudes, grad_raster_time=system.grad_raster_time
        )
        grad = make_arbitrary_grad(
            channel=channel,
            waveform=waveform,
            system=system,
            max_slew=max_slew,
            max_grad=max_grad,
            delay=times[0],
        )
    else:
        #  Keep the original possibly irregular sampling
        if np.any(
            np.abs(
                np.round(times / system.grad_raster_time) * system.grad_raster_time
                - times
            )
            > eps
        ):
            raise ValueError(
                'All time points must be on a gradient raster or "convert_to_arbitrary" option must be used.'
            )

        grad = SimpleNamespace()
        grad.type = "grad"
        grad.channel = channel
        grad.waveform = amplitudes
        grad.delay = (
            np.round(times[0] / system.grad_raster_time) * system.grad_raster_time
        )
        grad.tt = times - grad.delay
        grad.shape_dur = (
            np.round(times[-1] / system.grad_raster_time) * system.grad_raster_time
        )

    grad.first = amplitudes[0]
    grad.last = amplitudes[-1]

    return grad
