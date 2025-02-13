from types import SimpleNamespace
from typing import List, Union

import numpy as np

from pypulseq.add_gradients import add_gradients
from pypulseq.opts import Opts
from pypulseq.scale_grad import scale_grad
from pypulseq.utils.tracing import trace, trace_enabled


def __get_grad_abs_mag(grad: SimpleNamespace) -> np.ndarray:
    if grad.type == 'trap':
        return abs(grad.amplitude)
    return np.max(np.abs(grad.waveform))


def rotate3D(
    *args: SimpleNamespace, rotation_matrix: np.ndarray[np.float64], system: Union[Opts, None] = None
) -> List[SimpleNamespace]:
    """
    Rotates the corresponding gradient(s) by the provided rotation matrix. Non-gradient(s) are not affected.

    See also `pypulseq.rotate.rotate()` and `pypulseq.Sequence.sequence.add_block()`.

    Parameters
    ----------
    args : SimpleNamespace
        Gradient(s).
    rotation_matrix : np.ndarray[np.float64]
        3x3 rotation matrix by which the gradient(s) are rotated.
    system : Opts, default=Opts()
        System limits.

    Returns
    -------
    rotated_grads : [SimpleNamespace]
        Rotated gradient(s).
    """
    if system is None:
        system = Opts.default

    if rotation_matrix.shape != (3, 3):
        raise ValueError('The rotation matrix must have shape (3, 3).')

    # First create indexes of the objects to be bypassed or rotated
    axes = ['x', 'y', 'z']
    events_to_rotate_dict = {}
    i_bypass = []

    for i in range(len(args)):
        event = args[i]
        if event.type != 'grad' and event.type != 'trap':
            i_bypass.append(i)
        else:
            if event.channel not in axes:
                raise ValueError('Invalid event channel. Expected one of ' + str(axes))
            elif event.channel in events_to_rotate_dict:
                raise ValueError('More than one gradient for the same channel provided, channel: ' + str(event.channel))
            else:
                events_to_rotate_dict[event.channel] = event

    # Measure of relevant amplitude
    max_mag = 0
    for axis in axes:
        if axis in events_to_rotate_dict:
            event = events_to_rotate_dict[axis]
            max_mag = max(max_mag, __get_grad_abs_mag(event))
    fthresh = 1e-6
    thresh = fthresh * max_mag

    # Rotate the events (gradients)
    rotated_gradients = []
    for j in range(3):
        grad_out_curr = None
        for i in range(3):
            if axes[i] not in events_to_rotate_dict or abs(rotation_matrix[j, i]) < fthresh:
                continue
            scaled_gradient = scale_grad(grad=events_to_rotate_dict[axes[i]], scale=rotation_matrix[j, i])
            scaled_gradient.channel = axes[j]
            if grad_out_curr is None:
                grad_out_curr = scaled_gradient
            else:
                grad_out_curr = add_gradients((grad_out_curr, scaled_gradient), system=system)
        if grad_out_curr is not None and __get_grad_abs_mag(grad_out_curr) >= thresh:
            rotated_gradients.append(grad_out_curr)

    # Return
    bypass = np.take(args, i_bypass)
    return_grads = [*bypass, *rotated_gradients]

    if trace_enabled():
        for grad in return_grads:
            grad.trace = trace()

    return return_grads
