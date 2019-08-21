import numpy as np

from pypulseq.calc_ramp import calc_ramp
from pypulseq.opts import Opts
from copy import copy
from types import SimpleNamespace


def add_ramps(k, system=Opts(), rf: SimpleNamespace = None, max_grad: int = 0, max_slew: int = 0) -> list:
    """
    Adds segment so that `k` k-space trajectory ramps up from 0 to `k[0]` and ramps down from `k[-1]` to 0. If `k` is a
    tuple or list of k-space trajectories, ramp ups and ramp downs are added to each.

    Parameters
    ----------
    k : tuple, list or numpy.ndarray
        Tuple, list or numpy.ndarray of k-space trajectories to add ramp ups and ramp downs to.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    rf : SimpleNamespace, optional
        Zeros are added to this pulse sequence event over the ramp times in `k`.
    max_grad : int, optional
        Maximum gradient amplitude. Default is 0.
    max_slew : int, optional
        Maximum slew rate. Default is 0.

    Returns
    -------
    result : list
        List of ramped up and ramped down k-space trajectories from `k`.
    """
    if not isinstance(k, tuple) and not isinstance(k, list) and not isinstance(k, np.ndarray):
        raise Exception()

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
        raise Exception('Failed to calculate gradient ramps')

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
