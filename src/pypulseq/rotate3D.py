import copy
from types import SimpleNamespace
from typing import Any, List, Optional, Union

import numpy as np

from pypulseq.add_gradients import add_gradients
from pypulseq.opts import Opts
from pypulseq.scale_grad import scale_grad


def __get_grad_abs_mag(grad: SimpleNamespace) -> float:
    """Magnitude used for thresholding output components (MATLAB getGradAbsMag)."""
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
    MATLAB: mr.aux.quat.toRotMat(rotation)
    """
    q = np.asarray(q_wxyz, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError('Quaternion must have length 4 [w, x, y, z].')

    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError('Quaternion norm must be non-zero.')
    # normalize (MATLAB expects unit quaternion; we tolerate small numeric drift)
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _parse_rotation_to_matrix(
    *,
    rotation: Optional[Any] = None,
    rotation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    MATLAB parity:
      - rotation can be 3x3 matrix OR quaternion length-4 [w,x,y,z]
    Backward compatibility:
      - rotation_matrix keyword (your current API) still works.
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
    rotation: Optional[Any] = None,
    rotation_matrix: Optional[np.ndarray] = None,
    system: Union[Opts, None] = None,
) -> List[SimpleNamespace]:
    """
    Rotate gradient events (trap/grad) by a 3x3 rotation matrix (or quaternion like MATLAB).
    Non-gradient events (rf/adc/delay/etc) are passed through unchanged.

    Accepts mixed inputs like:
        ms_rotate3D(gx, gy, rf, adc, rotation_matrix=R, system=sys)
        ms_rotate3D(gx, gy, rf, adc, rotation=R, system=sys)
        ms_rotate3D(gx, gy, rf, adc, rotation=[w,x,y,z], system=sys)  # quaternion

    Also accepts list inputs:
        ms_rotate3D([gx, gy, gz], rf, adc, rotation_matrix=R, system=sys)
    """
    # ---- flatten list/tuple inputs inside args ----
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

    # ---- split into gradients vs bypass ----
    for event in args:
        if not hasattr(event, 'type') or event.type not in ('grad', 'trap'):
            bypass.append(event)
            continue

        if not hasattr(event, 'channel') or event.channel not in axes:
            raise ValueError(f'Invalid or missing gradient channel: {getattr(event, "channel", None)}')

        if event.channel in events_to_rotate:
            raise ValueError(f'More than one gradient for channel: {event.channel}')

        # deepcopy to avoid mutating caller objects (matching your current code)
        events_to_rotate[event.channel] = copy.deepcopy(event)

    # nothing to rotate -> just return bypass
    if len(events_to_rotate) == 0:
        return list(bypass)

    # ---- thresholding like MATLAB ----
    max_mag = 0.0
    for ax in axes:
        if ax in events_to_rotate:
            max_mag = max(max_mag, __get_grad_abs_mag(events_to_rotate[ax]))
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
                # raster-align before summing (your current approach)
                grad_out_curr = align_gradient_to_raster(grad_out_curr, system.grad_raster_time)
                scaled = align_gradient_to_raster(scaled, system.grad_raster_time)

                # add_gradients expects a list
                grad_out_curr = add_gradients([grad_out_curr, scaled], system=system)

        # only output non-zero amplitude gradients
        if grad_out_curr is not None and __get_grad_abs_mag(grad_out_curr) >= thresh:
            rotated_grads.append(grad_out_curr)

    return [*bypass, *rotated_grads]
