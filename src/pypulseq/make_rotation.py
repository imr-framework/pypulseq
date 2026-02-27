from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R


def make_rotation(*args) -> SimpleNamespace:
    """
    Create a rotation event to instruct the interpreter to rotate
    the gx, gy and gz waveforms according to the given rotation matrix.

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
    rot_quaternion : Rotation
        Scipy rotation operator. Name of the attribute is 'rot_quaternion'
        to align with MATLAB toolbox.

    Returns
    -------
    rotation : SimpleNamespace
        Rotation event.
    """
    if len(args) < 1:
        raise ValueError('Must supply rotation angle(s)')
    elif len(args) == 1:  # make_rotation(phi), make_rotation(rot_mat) or make_rotation(quaternion)
        if isinstance(args[0], float):  # make_rotation(phi)
            phi = float(args[0])
            if not (0 <= abs(phi) < 2 * np.pi):
                raise ValueError(f'Rotation angle phi ({phi:.2f}) is invalid. Should be in [0, 2π)')
            rot = R.from_rotvec(args[0] * np.asarray([0, 0, 1]))
        elif isinstance(args[0], (list, np.ndarray)) and args[0].shape == (3, 3):  # make_rotation(rot_mat)
            rot = R.from_matrix(np.asarray(args[0]))
        elif isinstance(args[0], (list, np.ndarray)) and args[0].shape == (4,):  # make_rotation(quaternion)
            quat = np.asarray(args[0], dtype=float)
            norm = np.linalg.norm(quat)
            quat = np.divide(quat, norm, where=norm > 0)
            rot = R.from_quat(quat, scalar_first=True)  # ensure unit quaternion
        else:
            raise ValueError('Unexpected input to make_rotation')
    elif len(args) == 2:  # make_rotation(phi, theta) or make_rotation(axis, angle)
        if isinstance(args[0], float):  # make_rotation(phi, theta)
            phi = float(args[0])
            theta = float(args[1]) if len(args) > 1 else 0.0
            if not (0 <= abs(phi) < 2 * np.pi):
                raise ValueError(f'Rotation angle phi ({phi:.2f}) is invalid. Should be in [0, 2π)')
            if not (0 <= abs(theta) <= np.pi):
                raise ValueError(f'Rotation angle theta ({theta:.2f}) is invalid. Should be in [0, π]')
            # First rotate about X (theta), then Z (phi)
            q1 = R.from_rotvec(theta * np.asarray([1, 0, 0]))  # theta: elevation
            q2 = R.from_rotvec(phi * np.asarray([0, 0, 1]))  # phi: azimuth
            rot = q1 * q2  # Apply q2 after q1 (q2 rotates in q1's frame)
        elif isinstance(args[0], (list, np.ndarray)) and len(args[0]) == 3:  # make_rotation(axis, angle)
            axis = np.asarray(args[0], dtype=float)
            phi = float(args[1])
            if not (0 <= abs(phi) <= np.pi):
                raise ValueError(f'Rotation angle phi ({phi:.2f}) is invalid. Should be in [0, π]')
            axis /= np.linalg.norm(axis)
            rot = R.from_rotvec(phi * axis)
        else:
            raise ValueError('Unexpected input to make_rotation')
    else:
        raise ValueError('Unexpected input to make_rotation')

    return SimpleNamespace(type='rot3D', rot_quaternion=rot)
