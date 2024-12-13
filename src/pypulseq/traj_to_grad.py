from typing import Tuple, Union

import numpy as np

from pypulseq.opts import Opts


def traj_to_grad(k: np.ndarray, raster_time: Union[float, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert k-space trajectory `k` into gradient waveform in compliance with `raster_time` gradient raster time.

    Parameters
    ----------
    k : numpy.ndarray
        K-space trajectory to be converted into gradient waveform.
    raster_time : float, default=Opts().grad_raster_time
        Gradient raster time.

    Returns
    -------
    g : numpy.ndarray
        Gradient waveform.
    sr : numpy.ndarray
        Slew rate.
    """
    if raster_time is None:
        raster_time = Opts.default.grad_raster_time

    # Compute finite difference for gradients in Hz/m
    g = (k[..., 1:] - k[..., :-1]) / raster_time
    # Compute the slew rate
    sr0 = (g[..., 1:] - g[..., :-1]) / raster_time

    # Gradient is now sampled between k-space points whilst the slew rate is between gradient points
    sr = np.zeros(sr0.shape[:-1] + (sr0.shape[-1] + 1,))
    sr[..., 0] = sr0[..., 0]
    sr[..., 1:-1] = 0.5 * (sr0[..., :-1] + sr0[..., 1:])
    sr[..., -1] = sr0[..., -1]

    return g, sr
