import math
from types import SimpleNamespace

import numpy as np
import pypulseq
import pytest
from pypulseq import rotate, rotate3D


def get_rotation_matrix(channel, angle_rad):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    if channel == 'x':
        rotation_matrix = [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]
    elif channel == 'y':
        rotation_matrix = [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
    elif channel == 'z':
        rotation_matrix = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
    else:
        raise ValueError('Channel must be "x", "y" or "z".')
    return np.array(rotation_matrix, dtype='float64')


def compare_gradient_sets(
    grad_set_A: list[SimpleNamespace],
    grad_set_B: list[SimpleNamespace],
    tolerance: float = 0,
) -> bool:
    """
    Compare two sets of gradients for equality. Each set may contain up to three gradients.
    The gradients must have channel 'x', 'y' or 'z'. In each set there must not be two gradients with the same channel.
    Allow a tolerance for numeric values.

    Parameters
    ----------
    grad_set_A : list[SimpleNamespace]
        The first set of gradients to compare. Must have channel 'x', 'y' or 'z'. Must not have two gradients with the same channel.
    grad_set_B : list[SimpleNamespace]
        The second set of gradients to compare. Must have channel 'x', 'y' or 'z'. Must not have two gradients with the same channel.
    tolerance : float = (default) 0
        The tolerance to allow for comparing numeric values.

    Returns
    -------
    is_equal : bool
        True if the two sets of gradients are equal with the tolerance.
    """
    channel_list = ['x', 'y', 'z']

    def check_gradient_set_and_get_channel_grad_dict(gradient_set):
        channel_grad_dict = {}
        assert len(gradient_set) <= 3, 'Each gradient set must not have more than three gradients.'
        for grad in gradient_set:
            assert hasattr(grad, 'channel'), 'Gradients must have attribute "channel".'
            assert grad.channel in channel_list, 'Gradients must have channel "x", "y" or "z".'
            assert grad.channel not in channel_grad_dict, (
                'There must not be two gradients with the same channel in each set.'
            )
            channel_grad_dict[grad.channel] = grad
        return channel_grad_dict

    channel_grad_dict_A = check_gradient_set_and_get_channel_grad_dict(grad_set_A)
    channel_grad_dict_B = check_gradient_set_and_get_channel_grad_dict(grad_set_B)

    def compare_gradients(grad_A, grad_B):
        grad_A_dict = grad_A.__dict__
        grad_B_dict = grad_B.__dict__
        if grad_A_dict.keys() != grad_B_dict.keys():
            return False

        for key, val_A in grad_A_dict.items():
            val_B = grad_B_dict[key]

            if isinstance(val_A, (float, np.float64)) and isinstance(val_B, (float, np.float64)):
                if abs(val_A - val_B) > tolerance:
                    return False
            elif isinstance(val_A, np.ndarray) and isinstance(val_B, np.ndarray):
                if val_A.dtype in (np.float64, np.float32) and val_B.dtype in (
                    np.float64,
                    np.float32,
                ):  # check if they are float arrays
                    if not np.allclose(
                        val_A, val_B, atol=tolerance, rtol=0
                    ):  # Using rtol=0 for pure absolute tolerance
                        return False
                elif val_A.shape != val_B.shape or (val_A != val_B).any():  # For non-float arrays or if shapes differ
                    return False
            elif val_A != val_B:
                return False
        return True

    for channel in channel_list:
        if (channel in channel_grad_dict_A) != (channel in channel_grad_dict_B):
            return False
        if channel in channel_grad_dict_A:
            grad_A = channel_grad_dict_A[channel]
            grad_B = channel_grad_dict_B[channel]
            if compare_gradients(grad_A, grad_B) == False:
                return False

    return True


angle_deg_list = [0.0, 0.1, 1, 60, 90, 180, 360, 400.1, -0.1, -1, -90, -180, -360]

grad_list = [
    pypulseq.make_trapezoid(channel='x', amplitude=1, duration=13),
    pypulseq.make_trapezoid(channel='y', amplitude=1, duration=13),
    pypulseq.make_trapezoid(channel='z', amplitude=1, duration=13),
    pypulseq.make_trapezoid(channel='x', amplitude=2, duration=5),
    pypulseq.make_trapezoid(channel='y', amplitude=2, duration=5),
    pypulseq.make_trapezoid(channel='z', amplitude=2, duration=5),
    pypulseq.make_extended_trapezoid('x', [0, 5, 1, 3], convert_to_arbitrary=True, times=[1, 3, 4, 7]),
    pypulseq.make_extended_trapezoid('y', [0, 5, 1, 3], convert_to_arbitrary=True, times=[1, 3, 4, 7]),
    pypulseq.make_extended_trapezoid('z', [0, 5, 1, 3], convert_to_arbitrary=True, times=[1, 3, 4, 7]),
    pypulseq.make_extended_trapezoid('x', [0, 5, 1, 3], convert_to_arbitrary=False, times=[1, 3, 4, 7]),
    pypulseq.make_extended_trapezoid('y', [0, 5, 1, 3], convert_to_arbitrary=False, times=[1, 3, 4, 7]),
    pypulseq.make_extended_trapezoid('z', [0, 5, 1, 3], convert_to_arbitrary=False, times=[1, 3, 4, 7]),
    pypulseq.make_extended_trapezoid('x', [0, 3, 2, 3], convert_to_arbitrary=False, times=[1, 2, 3, 4]),
    pypulseq.make_extended_trapezoid('y', [0, 3, 2, 3], convert_to_arbitrary=False, times=[1, 2, 3, 4]),
    pypulseq.make_extended_trapezoid('z', [0, 3, 2, 3], convert_to_arbitrary=False, times=[1, 2, 3, 4]),
]


@pytest.mark.filterwarnings('ignore:When using rotate():UserWarning')
@pytest.mark.parametrize('angle_deg', angle_deg_list)
def test_rotation3D_vs_rotation(angle_deg):
    """Compare results of rotate and rotate3D."""

    channel_list = ['x', 'y', 'z']
    angle_rad = angle_deg * math.pi / 180

    for rotation_axis in channel_list:
        rotation_matrix = get_rotation_matrix(rotation_axis, angle_rad)

        for grad in grad_list:
            grads_rotated = rotate(*[grad], angle=angle_rad, axis=rotation_axis)
            grads_rotated3D = rotate3D(*[grad], rotation_matrix=rotation_matrix)

            assert compare_gradient_sets(grads_rotated, grads_rotated3D, tolerance=1e-4), (
                f'Result of rotate and rotate3D should be the same! Angle: {angle_deg}, Axis: {rotation_axis}, Grad: {grad}'
            )


@pytest.mark.filterwarnings('ignore:When using rotate():UserWarning')
@pytest.mark.parametrize('angle_deg', angle_deg_list)
def test_rotation3D_vs_rotation_double(angle_deg):
    """Compare results of rotate and rotate3D."""

    channel_list = ['x', 'y', 'z']
    angle_rad = angle_deg * math.pi / 180

    for rotation_axis in channel_list:
        rotation_matrix = get_rotation_matrix(rotation_axis, angle_rad)

        for grad in grad_list:
            grads_rotated = rotate(*[grad], angle=angle_rad, axis=rotation_axis)
            grads_rotated3D = rotate3D(*[grad], rotation_matrix=rotation_matrix)

            grads_rotated_double = rotate(*grads_rotated, angle=angle_rad, axis=rotation_axis)
            grads_rotated3D_double = rotate3D(*grads_rotated3D, rotation_matrix=rotation_matrix)

            assert compare_gradient_sets(grads_rotated_double, grads_rotated3D_double, tolerance=1e-4), (
                f'Result of double rotate and rotate3D should be the same! Angle: {angle_deg}, Axis: {rotation_axis}, Grad: {grad}'
            )


@pytest.mark.filterwarnings('ignore:When using rotate():UserWarning')
@pytest.mark.parametrize('angle_deg', angle_deg_list)
def test_rotation3D_vs_rotation_double_2(angle_deg):
    """Compare results of rotate and rotate3D."""

    channel_list = ['x', 'y', 'z']
    angle_rad = angle_deg * math.pi / 180

    for rotation_axis in channel_list:
        rotation_matrix = get_rotation_matrix(rotation_axis, angle_rad)

        for grad in grad_list:
            grads_rotated = rotate(*[grad], angle=angle_rad, axis=rotation_axis)

            grads_rotated_double = rotate(*grads_rotated, angle=angle_rad, axis=rotation_axis)
            grads_rotated3D_double_2 = rotate3D(*[grad], rotation_matrix=rotation_matrix @ rotation_matrix)

            assert compare_gradient_sets(grads_rotated_double, grads_rotated3D_double_2, tolerance=1e-4), (
                f'Result of second double rotate and rotate3D should be the same! Angle: {angle_deg}, Axis: {rotation_axis}, Grad: {grad}'
            )
