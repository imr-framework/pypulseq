import numpy as np

from pypulseq.opts import Opts


def traj_to_grad(k: np.ndarray, raster_time: float = Opts().grad_raster_time):
    """
    Convert k-space trajectory `k` into gradient waveform in compliance with `raster_time` gradient raster time.

    Parameters
    ----------
    k : numpy.ndarray
        K-space trajectory to be converted into gradient waveform.
    raster_time : float, optional
        Gradient raster time. Default is the default value of gradient raster time set in system limits `Opts`.

    Returns
    -------
    g : numpy.ndarray
        Gradient waveform.
    sr : numpy.ndarray
        Slew rate.
    """
    g = (k[1:] - k[:-1]) / raster_time
    sr0 = (g[1:] - g[:-1]) / raster_time
    sr = np.zeros(len(sr0) + 1)
    sr[0] = sr0[0]
    sr[1:-1] = 0.5 * (sr0[-1] + sr0[1:])
    sr[-1] = sr0[-1]

    return g, sr
