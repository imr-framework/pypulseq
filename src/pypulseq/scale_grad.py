from copy import copy
from types import SimpleNamespace
from typing import Union

import numpy as np

from pypulseq import eps
from pypulseq.opts import Opts


def scale_grad(grad: SimpleNamespace, scale: float, system: Union[Opts, None] = None) -> SimpleNamespace:
    """
    Scales the gradient with the scalar.

    Parameters
    ----------
    grad : SimpleNamespace
        Gradient event to be scaled.
    scale : float
        Scaling factor.
    system : Opts, default=Opts()
        System limits.

    Returns
    -------
    grad : SimpleNamespace
        Scaled gradient.
    """
    # copy() to emulate pass-by-value; otherwise passed grad event is modified
    scaled_grad = copy(grad)
    if scaled_grad.type == 'trap':
        scaled_grad.amplitude = scaled_grad.amplitude * scale
        scaled_grad.flat_area = scaled_grad.flat_area * scale
        if system is not None:
            if abs(scaled_grad.amplitude) > system.max_grad:
                raise ValueError(
                    f'scale_grad: maximum amplitude exceeded {100 * abs(scaled_grad.amplitude) / system.max_grad} %'
                )
            if (
                abs(grad.amplitude) > eps
                and abs(scaled_grad.amplitude) / min(scaled_grad.rise_time, scaled_grad.fall_time) > system.max_slew
            ):
                raise ValueError(
                    'mr.scale_grad: maximum slew rate exceeded {100 * abs(scaled_grad.amplitude) / min(scaled_grad.rise_time, scaled_grad.fall_time) / system.max_slew} %'
                )

    else:
        scaled_grad.waveform = scaled_grad.waveform * scale
        scaled_grad.first = scaled_grad.first * scale
        scaled_grad.last = scaled_grad.last * scale
        if system is not None:
            if max(abs(scaled_grad.waveform)) > system.max_grad:
                raise ValueError(
                    f'scale_grad: maximum amplitude exceeded {100 * max(abs(scaled_grad.waveform)) / system.max_grad} %'
                )
            if max(abs(scaled_grad.waveform)) > eps:
                scaled_grad_max_abs_slew = max(abs(np.diff(scaled_grad.waveform) / np.diff(grad.tt)))
                if scaled_grad_max_abs_slew > system.max_slew:
                    raise ValueError(
                        f'scale_grad: maximum slew rate exceeded {100 * scaled_grad_max_abs_slew / system.max_slew} %'
                    )
    scaled_grad.area = scaled_grad.area * scale

    if hasattr(scaled_grad, 'id'):
        delattr(scaled_grad, 'id')

    return scaled_grad
