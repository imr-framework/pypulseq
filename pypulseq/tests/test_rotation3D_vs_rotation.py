import math
from types import SimpleNamespace

import numpy as np

import pypulseq
from pypulseq import rotate, rotate3D


def compare_gradient_sets(
    grad_set_A: list[SimpleNamespace], grad_set_B: list[SimpleNamespace], tolerance: float = 0
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
            assert (
                grad.channel not in channel_grad_dict
            ), 'There must not be two gradients with the same channel in each set.'
            channel_grad_dict[grad.channel] = grad
        return channel_grad_dict

    channel_grad_dict_A = check_gradient_set_and_get_channel_grad_dict(grad_set_A)
    channel_grad_dict_B = check_gradient_set_and_get_channel_grad_dict(grad_set_B)

    def compare_gradients(grad_A, grad_B):
        grad_A_dict = grad_A.__dict__
        grad_B_dict = grad_B.__dict__
        if len(grad_A_dict) != len(grad_B_dict):
            return False
        for key in grad_A_dict:
            if not key in grad_B_dict:
                return False
            elif (type(grad_A_dict[key]) == float and type(grad_B_dict[key]) == float) or (
                type(grad_A_dict[key]) == np.float64 and type(grad_B_dict[key]) == np.float64
            ):
                if grad_A_dict[key] - grad_B_dict[key] > tolerance:
                    return False
            elif type(grad_A_dict[key]) == np.ndarray and type(grad_B_dict[key]) == np.ndarray:
                if grad_A_dict[key].dtype == np.float64 and grad_B_dict[key].dtype == np.float64:
                    if (grad_A_dict[key] - grad_B_dict[key] > tolerance).any():
                        return False
                else:
                    if (grad_A_dict[key] != grad_B_dict[key]).any():
                        return False
            elif grad_A_dict[key] != grad_B_dict[key]:
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


def test_rotation3D_vs_rotation():
    """
    Create some trapezoids and extended trapezoids and compare the results of applying rotate and rotate3D.
    """
    print('---- test_rotation3D_vs_rotation() ----')

    def get_rotation_matrix(channel, angle_radians):
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        if channel == 'x':
            rotation_matrix = [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]
        elif channel == 'y':
            rotation_matrix = [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
        elif channel == 'z':
            rotation_matrix = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
        else:
            raise ValueError('Channel must be "x", "y" or "z".')
        return np.array(rotation_matrix, dtype='float64')

    channel_list = ['x', 'y', 'z']

    # prepare gradients
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

    for angle_degree in (
        [0.0, 0.1, 1, 2, 3, 60, 90, 180, 360, *list(range(10, 450, 30)), -0.1, -1, -90, -180, -360, -400]
    ):
        print('angle_degree:', angle_degree)
        angle_radians = angle_degree * math.pi / 180

        for rotation_axis in channel_list:
            rotation_matrix = get_rotation_matrix(rotation_axis, angle_radians)

            for grad in grad_list:
                # apply rotations
                grads_rotated = rotate(*[grad], angle=angle_radians, axis=rotation_axis)
                grads_rotated3D = rotate3D(*[grad], rotation_matrix=rotation_matrix)
                grads_rotated_double = rotate(*grads_rotated, angle=angle_radians, axis=rotation_axis)
                grads_rotated3D_double = rotate3D(*grads_rotated3D, rotation_matrix=rotation_matrix)
                grads_rotated3D_double_2 = rotate3D(*[grad], rotation_matrix=rotation_matrix @ rotation_matrix)

                # check results
                # print("------")
                # print("angle_degree:", angle_degree)
                # print("rotation_axis:", rotation_axis)
                # print("grads_rotated:", grads_rotated)
                # print("grads_rotated3D:", grads_rotated3D)
                # print("grads_rotated_double:", grads_rotated_double)
                # print("grads_rotated3D_double:", grads_rotated3D_double)
                # print("grads_rotated3D_double_2:", grads_rotated3D_double_2)
                assert compare_gradient_sets(
                    grads_rotated, grads_rotated3D, tolerance=1e-6
                ), 'Result of rotate and rotate3D should be the same!'
                assert compare_gradient_sets(
                    grads_rotated_double, grads_rotated3D_double, tolerance=1e-6
                ), 'Result of double rotate and rotate3D should be the same!'
                assert compare_gradient_sets(
                    grads_rotated_double, grads_rotated3D_double_2, tolerance=1e-6
                ), 'Result of second double rotate and rotate3D should be the same!'

    print('Tests ok.')


if __name__ == '__main__':
    test_rotation3D_vs_rotation()
