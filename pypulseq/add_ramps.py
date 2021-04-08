from copy import copy
from types import SimpleNamespace
from typing import Union, List

import numpy as np

from pypulseq.calc_ramp import calc_ramp
from pypulseq.opts import Opts


def add_ramps(k: Union[list, np.ndarray, tuple], system=Opts(), rf: SimpleNamespace = None, max_grad: int = 0,
              max_slew: int = 0) -> List[np.ndarray]:
    """
    Adds segment so that `k` k-space trajectory ramps up from 0 to `k[0]` and ramps down from `k[-1]` to 0. If `k` is a
    tuple or list of k-space trajectories, ramp-ups and ramp-downs are added to each.

    Parameters
    ----------
    k : array_like
        Array-like of k-space trajectories to add ramp-ups and -downs to.
    system : Opts, optional, default=Opts()
        System limits.
    rf : SimpleNamespace, optional
        Zeros are added to this pulse sequence event over the ramp times in `k`.
    max_grad : int, optional, default=0
        Maximum gradient amplitude.
    max_slew : int, optional, default=0
        Maximum slew rate.

    Returns
    -------
    result : list[
        List of ramped up and ramped down k-space trajectories from `k`.

    Raises
    ------
    ValueError
        If `k` is not list, np.ndarray or tuple
    RuntimeError
        If gradient ramps fail to be calculated
    """
    if not isinstance(k, (list, np.ndarray, tuple)):
        raise ValueError(f'k has to be one of list, np.ndarray, tuple. Passed: {type(k)}')

    k_arg = copy(k)
    if max_grad > 0:
        system.max_grad = max_grad

    if max_slew > 0:
        system.max_slew = max_slew

    k = np.vstack(k)
    num_channels = k.shape[0]
    k = np.vstack((k, np.zeros((3 - num_channels, k.shape[1]))))

    k_up, ok1 = calc_ramp(np.zeros((3, 2)), k[:, :2], system)
    k_down, ok2 = calc_ramp(k[:, -2:], np.zeros((3, 2)), system)
    if not (ok1 and ok2):
        raise RuntimeError('Failed to calculate gradient ramps')

    k_up = np.hstack((np.zeros((3, 2)), k_up))
    k_down = np.hstack((k_down, np.zeros((3, 1))))

    k = np.hstack((k_up, k, k_down))

    result = []
    if not isinstance(k_arg, list):
        result.append(k[:num_channels])
    else:
        for i in range(num_channels):
            result.append(k[i])

    if rf is not None:
        result.append(np.concatenate((np.zeros(k_up.shape[1] * 10), rf, np.zeros(k_down.shape[1] * 10))))

    return result
