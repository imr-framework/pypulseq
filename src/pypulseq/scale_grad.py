from copy import copy
from types import SimpleNamespace


def scale_grad(grad: SimpleNamespace, scale: float) -> SimpleNamespace:
    """
    Scales the gradient with the scalar.

    Parameters
    ----------
    grad : SimpleNamespace
        Gradient event to be scaled.
    scale : float
        Scaling factor.

    Returns
    -------
    grad : SimpleNamespace
        Scaled gradient.
    """
    # copy() to emulate pass-by-value; otherwise passed grad event is modified
    scaled_grad = copy(grad)
    if scaled_grad.type == 'trap':
        scaled_grad.amplitude = scaled_grad.amplitude * scale
        scaled_grad.area = scaled_grad.area * scale
        scaled_grad.flat_area = scaled_grad.flat_area * scale
    else:
        scaled_grad.waveform = scaled_grad.waveform * scale
        scaled_grad.first = scaled_grad.first * scale
        scaled_grad.last = scaled_grad.last * scale

    if hasattr(scaled_grad, 'id'):
        delattr(scaled_grad, 'id')

    return scaled_grad
