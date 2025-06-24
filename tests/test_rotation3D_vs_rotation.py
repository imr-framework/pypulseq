import numpy as np
import pypulseq
import pytest
from pypulseq import rotate, rotate3D

from conftest import Approx, get_rotation_matrix

channel_list = ['x', 'y', 'z']
angle_deg_list = [0.0, 0.1, 1.0, 60.0, 90.0, 180.0, 360.0, 400.1, -0.1, -1.0, -90.0, -180.0, -360.0]

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


def __list_to_dict(gradient_set):
    channel_grad_dict = {}
    assert len(gradient_set) <= 3, 'Each gradient set must not have more than three gradients.'
    for grad in gradient_set:
        assert grad.channel in channel_list, 'Gradients must have channel "x", "y" or "z".'
        assert grad.channel not in channel_grad_dict, (
            'There must not be two gradients with the same channel in each set.'
        )
        channel_grad_dict[grad.channel] = grad
    return channel_grad_dict


@pytest.mark.filterwarnings('ignore:When using rotate():UserWarning')
@pytest.mark.parametrize('angle_deg', angle_deg_list)
def test_rotation3D_vs_rotation(angle_deg):
    """Compare results of rotate and rotate3D."""
    angle_rad = np.deg2rad(angle_deg)

    for rotation_axis in channel_list:
        rotation_matrix = get_rotation_matrix(rotation_axis, angle_rad)

        for grad in grad_list:
            grads_rotated = __list_to_dict(rotate(grad, angle=angle_rad, axis=rotation_axis))
            grads_rotated3D = __list_to_dict(rotate3D(grad, rotation_matrix=rotation_matrix))

            assert grads_rotated3D == Approx(grads_rotated, abs=1e-4, rel=1e-4), (
                f'Result of rotate and rotate3D should be the same! Angle: {angle_deg}, Axis: {rotation_axis}, Grad: {grad}'
            )


@pytest.mark.filterwarnings('ignore:When using rotate():UserWarning')
@pytest.mark.parametrize('angle_deg', angle_deg_list)
def test_rotation3D_vs_rotation_double(angle_deg):
    """Compare results of rotate and rotate3D."""
    angle_rad = np.deg2rad(angle_deg)

    for rotation_axis in channel_list:
        rotation_matrix = get_rotation_matrix(rotation_axis, angle_rad)

        for grad in grad_list:
            grads_rotated = rotate(grad, angle=angle_rad, axis=rotation_axis)
            grads_rotated_double = __list_to_dict(rotate(*grads_rotated, angle=angle_rad, axis=rotation_axis))

            grads_rotated3D = rotate3D(grad, rotation_matrix=rotation_matrix)
            grads_rotated3D_double = __list_to_dict(rotate3D(*grads_rotated3D, rotation_matrix=rotation_matrix))

            assert grads_rotated3D_double == Approx(grads_rotated_double, abs=1e-4, rel=1e-4), (
                f'Result of double rotate and rotate3D should be the same! Angle: {angle_deg}, Axis: {rotation_axis}, Grad: {grad}'
            )


@pytest.mark.filterwarnings('ignore:When using rotate():UserWarning')
@pytest.mark.parametrize('angle_deg', angle_deg_list)
def test_rotation3D_vs_rotation_double_2(angle_deg):
    """Compare results of rotate and rotate3D."""
    # print("Two steps vs single step")
    angle_rad = np.deg2rad(angle_deg)

    for rotation_axis in channel_list:
        rotation_matrix = get_rotation_matrix(rotation_axis, angle_rad)

        for grad in grad_list:
            # print(f'Rotating about {rotation_axis} axis by {angle_deg} degrees')
            grads_rotated = rotate(grad, angle=angle_rad, axis=rotation_axis)
            grads_rotated_double = __list_to_dict(rotate(*grads_rotated, angle=angle_rad, axis=rotation_axis))

            grads_rotated3D_double = __list_to_dict(rotate3D(grad, rotation_matrix=rotation_matrix @ rotation_matrix))

            for g in grads_rotated_double.values():
                print(f'two steps {g.channel} area: {g.area}')
            for g in grads_rotated3D_double.values():
                print(f'single step {g.channel} area: {g.area}')
            assert grads_rotated3D_double == Approx(grads_rotated_double, abs=1e-4, rel=1e-4), (
                f'Result of second double rotate and rotate3D should be the same! Angle: {angle_deg}, Axis: {rotation_axis}, Grad: {grad}'
            )
