from types import SimpleNamespace
from typing import Tuple

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts


def split_gradient(
    grad: SimpleNamespace, system: Opts = Opts()
) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """
    Splits a trapezoidal gradient into slew up, flat top and slew down. Returns the individual gradient parts (slew up,
    flat top and slew down) as extended trapezoid gradient objects. The delays in the individual gradient events are
    adapted such that addGradients(...) produces an gradient equivalent to 'grad'.

    See also:
    - `pypulseq.split_gradient()`
    - `pypulseq.make_extended_trapezoid()`
    - `pypulseq.make_trapezoid()`
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.opts.Opts`

    Parameters
    ----------
    grad : SimpleNamespace
        Gradient event to be split into two gradient waveforms.
    system : Opts, default=Opts()
        System limits.

    Returns
    -------
    grad1, grad2 : SimpleNamespace
        Split gradient waveforms.

    Raises
    ------
    ValueError
         If arbitrary gradients are passed.
         If non-gradient event is passed.
    """
    grad_raster_time = system.grad_raster_time
    total_length = calc_duration(grad)

    if grad.type == "trap":
        channel = grad.channel
        grad.delay = np.round(grad.delay / grad_raster_time) * grad_raster_time
        grad.rise_time = np.round(grad.rise_time / grad_raster_time) * grad_raster_time
        grad.flat_time = np.round(grad.flat_time / grad_raster_time) * grad_raster_time
        grad.fall_time = np.round(grad.fall_time / grad_raster_time) * grad_raster_time

        times = np.array([0, grad.rise_time])
        amplitudes = np.array([0, grad.amplitude])
        ramp_up = make_extended_trapezoid(
            channel=channel,
            system=system,
            times=times,
            amplitudes=amplitudes,
            skip_check=True,
        )
        ramp_up.delay = grad.delay

        times = np.array([0, grad.fall_time])
        amplitudes = np.array([grad.amplitude, 0])
        ramp_down = make_extended_trapezoid(
            channel=channel,
            system=system,
            times=times,
            amplitudes=amplitudes,
            skip_check=True,
        )
        ramp_down.delay = total_length - grad.fall_time
        ramp_down.t = ramp_down.t * grad_raster_time

        flat_top = SimpleNamespace()
        flat_top.type = "grad"
        flat_top.channel = channel
        flat_top.delay = grad.delay + grad.rise_time
        flat_top.t = np.arange(
            step=grad_raster_time,
            stop=ramp_down.delay - grad_raster_time - grad.delay - grad.rise_time,
        )
        flat_top.waveform = grad.amplitude * np.ones(len(flat_top.t))
        flat_top.first = grad.amplitude
        flat_top.last = grad.amplitude

        return ramp_up, flat_top, ramp_down
    elif grad.type == "grad":
        raise ValueError("Splitting of arbitrary gradients is not implemented yet.")
    else:
        raise ValueError("Splitting of unsupported event.")
