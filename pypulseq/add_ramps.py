from copy import copy
from types import SimpleNamespace
from typing import Union, List

import numpy as np

from pypulseq.calc_ramp import calc_ramp
from pypulseq.opts import Opts


def add_ramps(
    k: Union[list, np.ndarray, tuple],
    max_grad: int = 0,
    max_slew: int = 0,
    rf: SimpleNamespace = None,
    system=Opts(),
) -> List[np.ndarray]:
    """
    Add segments to the trajectory to ramp to and from the given trajectory.

    Parameters
    ----------
    k : numpy.ndarray, or [numpy.ndarray, ...]
        If `k` is a single trajectory: Add a segment to `k` so `k_out` travels from 0 to `k[0]` and a segment so `k_out`
        goes from `k[-1]` back to 0 without violating the gradient and slew constraints.
        If `k` is multiple trajectoriess: add segments of the same length for each trajectory in the cell array.
    system : Opts, default=Opts()
        System limits.
    rf : SimpleNamespace, default=None
        Add a segment of zeros over the ramp times to an RF shape.
    max_grad : int, default=0
        Maximum gradient amplitude.
    max_slew : int, default=0
        Maximum slew rate.

    Returns
    -------
    result : [numpy.ndarray, ...]
        List of ramped up and ramped down k-space trajectories from `k`.

    Raises
    ------
    ValueError
        If `k` is not list, np.ndarray or tuple
    RuntimeError
        If gradient ramps fail to be calculated
    """
    if not isinstance(k, (list, np.ndarray, tuple)):
        raise ValueError(
            f"k has to be one of list, np.ndarray, tuple. Passed: {type(k)}"
        )

    k_arg = copy(k)
    if max_grad > 0:
        system.max_grad = max_grad

    if max_slew > 0:
        system.max_slew = max_slew

    k = np.vstack(k)
    num_channels = k.shape[0]
    k = np.vstack(
        (k, np.zeros((3 - num_channels, k.shape[1])))
    )  # Pad with zeros if needed

    k_up, ok1 = calc_ramp(k0=np.zeros((3, 2)), k_end=k[:, :2], system=system)
    k_down, ok2 = calc_ramp(k0=k[:, -2:], k_end=np.zeros((3, 2)), system=system)
    if not (ok1 and ok2):
        raise RuntimeError("Failed to calculate gradient ramps")

    # Add start and end points to ramps
    k_up = np.hstack((np.zeros((3, 2)), k_up))
    k_down = np.hstack((k_down, np.zeros((3, 1))))

    # Add ramps to trajectory
    k = np.hstack((k_up, k, k_down))

    result = []
    if not isinstance(k_arg, list):
        result.append(k[:num_channels])
    else:
        for i in range(num_channels):
            result.append(k[i])

    if rf is not None:
        result.append(
            np.concatenate(
                (np.zeros(k_up.shape[1] * 10), rf, np.zeros(k_down.shape[1] * 10))
            )
        )

    return result
