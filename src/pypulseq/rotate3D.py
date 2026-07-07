import copy
from types import SimpleNamespace
from typing import Any, List, Union

import numpy as np
from scipy.spatial.transform import Rotation

from pypulseq.add_gradients import add_gradients
from pypulseq.opts import Opts
from pypulseq.scale_grad import scale_grad


def _get_grad_abs_mag(grad: SimpleNamespace) -> float:
    """Magnitude used for thresholding output components."""
    if grad.type == 'trap':
        return float(abs(grad.amplitude))
    return float(np.max(np.abs(grad.waveform)))


def align_gradient_to_raster(g: SimpleNamespace, raster: float) -> SimpleNamespace:
    """
    Align gradient timing parameters to the gradient raster and verify rounding error.
    This is used to avoid add_gradients() complaining about non-raster time points.
    """
    tol = 1e-9  # seconds

    if getattr(g, 'type', None) == 'grad':
        original_tt = np.array(g.tt, dtype=np.float64)
        rounded_tt = np.round(original_tt / raster) * raster
        if np.any(np.abs(original_tt - rounded_tt) > tol):
            raise ValueError(f"'grad' tt values not aligned to raster (>{tol}s): {original_tt} -> {rounded_tt}")
        g.tt = rounded_tt
        g.waveform = np.array(g.waveform, dtype=np.float64)

    elif getattr(g, 'type', None) == 'trap':
        original_delay = float(getattr(g, 'delay', 0.0))
        original_rise = float(getattr(g, 'rise_time', 0.0))
        original_flat = float(getattr(g, 'flat_time', 0.0))
        original_fall = float(getattr(g, 'fall_time', 0.0))

        rounded_delay = float(np.round(original_delay / raster) * raster)
        rounded_rise = float(np.round(original_rise / raster) * raster)
        rounded_flat = float(np.round(original_flat / raster) * raster)
        rounded_fall = float(np.round(original_fall / raster) * raster)

        if any(
            abs(x - y) > tol
            for x, y in [
                (original_delay, rounded_delay),
                (original_rise, rounded_rise),
                (original_flat, rounded_flat),
                (original_fall, rounded_fall),
            ]
        ):
            raise ValueError(
                f"'trap' timing values not aligned to raster (>{tol}s): "
                f'delay={original_delay}→{rounded_delay}, '
                f'rise={original_rise}→{rounded_rise}, '
                f'flat={original_flat}→{rounded_flat}, '
                f'fall={original_fall}→{rounded_fall}'
            )

        g.delay = rounded_delay
        g.rise_time = rounded_rise
        g.flat_time = rounded_flat
        g.fall_time = rounded_fall

    # always align delay if present
    original_delay = float(getattr(g, 'delay', 0.0))
    rounded_delay = float(np.round(original_delay / raster) * raster)
    if abs(original_delay - rounded_delay) > tol:
        raise ValueError(f"'delay' value not aligned to raster (>{tol}s): {original_delay} → {rounded_delay}")
    g.delay = rounded_delay

    return g


def _quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w,x,y,z] (scalar first) to a 3x3 rotation matrix.
    """
    q = np.asarray(q_wxyz, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError('Quaternion must have length 4 [w, x, y, z].')

    if np.linalg.norm(q) == 0:
        raise ValueError('Quaternion norm must be non-zero.')

    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def _parse_rotation_to_matrix(
    *,
    rotation: Union[Any, None] = None,
    rotation_matrix: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    rotation can be 3x3 matrix OR quaternion length-4 [w,x,y,z]
    """
    if rotation is None and rotation_matrix is None:
        raise ValueError("You must provide either 'rotation' (matrix or quaternion) or 'rotation_matrix' (3x3).")

    if rotation_matrix is not None:
        rot = np.asarray(rotation_matrix, dtype=float)
    else:
        rot = np.asarray(rotation, dtype=float)

    if rot.shape == (3, 3):
        return rot

    # quaternion case
    if rot.size == 4:
        return _quat_wxyz_to_rotmat(rot)

    raise ValueError('rotation must be either a 3x3 matrix or a quaternion [w, x, y, z] (scalar first).')


def rotate3D(
    *args: SimpleNamespace,
    rotation: Union[Any, None] = None,
    rotation_matrix: Union[np.ndarray, None] = None,
    system: Union[Opts, None] = None,
) -> List[SimpleNamespace]:
    """
    Rotate gradient events by a 3*3 rotation matrix or quaternion.

    Non-gradient events (e.g. RF, ADC, delay) are passed through unchanged.

    Parameters
    args : SimpleNamespace or list of SimpleNamespace
        Input events. Can include gradient events ('grad', 'trap') and
        non-gradient events (rf, adc, delay, etc.). Lists or tuples of events
        are also accepted and will be flattened.

    rotation : array_like or None, Union
        Rotation specified either as a 3*3 matrix or a quaternion
        [w, x, y, z] (scalar-first convention). Default is None.

    rotation_matrix : np.ndarray or None, Union
        Explicit 3*3 rotation matrix. Provided for backward compatibility.
        If both 'rotation' and 'rotation_matrix' are given, 'rotation_matrix'
        takes precedence. Default is None.

    system : Opts or None, Union
        PyPulseq system limits. If None, 'Opts.default' is used.

    Returns
    out : list of SimpleNamespace
        List of events including rotated gradient components and unchanged
        non-gradient events.

    Notes
    Only one gradient per axis (x, y, z) is allowed in the input.

    Examples
    >>> import numpy as np
    >>> import pypulseq as pp
    >>> from rotate3D import rotate3D
    >>>
    >>> system = pp.Opts()
    >>> gx = pp.make_trapezoid(channel='x', system=system, amplitude=100, duration=2e-3)
    >>> gy = pp.make_trapezoid(channel='y', system=system, amplitude=50, duration=2e-3)
    >>> gz = pp.make_trapezoid(channel="z", system=system, amplitude=20, duration=2e-3)
    >>> rf = pp.make_block_pulse(flip_angle=np.pi/2, duration=1e-3, system=system)
    >>> adc = pp.make_adc(num_samples=64, dwell=10e-6, delay=0, system=system)
    >>>
    >>> angle = np.pi / 4
    >>> Rz = np.array([
    ...     [np.cos(angle), -np.sin(angle), 0],
    ...     [np.sin(angle),  np.cos(angle), 0],
    ...     [0,              0,             1],
    ... ])
    >>>
    >>> q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], dtype=float)
    >>>
    >>> out1 = rotate3D(gx, gy, gz, rf, adc, rotation_matrix=Rz, system=system)
    >>> out2 = rotate3D(gx, gy, gz, rf, adc, rotation=Rz, system=system)
    >>> out3 = rotate3D(gx, gy, gz, rf, adc, rotation=q, system=system)
    """

    # flatten list/tuple inputs inside args
    flat_args: List[SimpleNamespace] = []
    for a in args:
        if a is None:
            continue
        if isinstance(a, (list, tuple)):
            for x in a:
                if x is not None:
                    flat_args.append(x)
        else:
            flat_args.append(a)
    args = tuple(flat_args)

    if system is None:
        system = Opts.default

    rotMat = _parse_rotation_to_matrix(rotation=rotation, rotation_matrix=rotation_matrix)

    axes = ['x', 'y', 'z']
    events_to_rotate: dict[str, SimpleNamespace] = {}
    bypass: List[SimpleNamespace] = []

    # split into gradients vs bypass
    for event in args:
        if not hasattr(event, 'type') or event.type not in ('grad', 'trap'):
            bypass.append(event)
            continue

        if not hasattr(event, 'channel') or event.channel not in axes:
            raise ValueError(f'Invalid or missing gradient channel: {getattr(event, "channel", None)}')

        if event.channel in events_to_rotate:
            raise ValueError(f'More than one gradient for channel: {event.channel}')

        events_to_rotate[event.channel] = copy.deepcopy(event)

    if len(events_to_rotate) == 0:
        return list(bypass)

    # thresholding
    max_mag = 0.0
    for ax in axes:
        if ax in events_to_rotate:
            max_mag = max(max_mag, _get_grad_abs_mag(events_to_rotate[ax]))
    fthresh = 1e-6
    thresh = fthresh * max_mag

    rotated_grads: List[SimpleNamespace] = []
    for j in range(3):
        grad_out_curr = None
        for i in range(3):
            ax_in = axes[i]
            if ax_in not in events_to_rotate:
                continue

            w = float(rotMat[j, i])
            if abs(w) < fthresh:
                continue

            scaled = scale_grad(grad=events_to_rotate[ax_in], scale=w)
            scaled.channel = axes[j]

            if grad_out_curr is None:
                grad_out_curr = scaled
            else:
                # grad_out_curr = align_gradient_to_raster(grad_out_curr, system.grad_raster_time)
                # scaled = align_gradient_to_raster(scaled, system.grad_raster_time)
                grad_out_curr = add_gradients([grad_out_curr, scaled], system=system)
                grad_out_curr = align_gradient_to_raster(grad_out_curr, system.grad_raster_time)

        if grad_out_curr is not None and _get_grad_abs_mag(grad_out_curr) >= thresh:
            rotated_grads.append(grad_out_curr)

    return [*bypass, *rotated_grads]
