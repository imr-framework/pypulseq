import numpy as np

from pypulseq.opts import Opts


def traj_to_grad(k, raster_time=Opts().grad_raster_time):
    g = (k[1:] - k[:-1]) / raster_time
    sr0 = (g[1:] - g[:-1]) / raster_time
    sr = np.zeros(len(sr0) + 1)
    sr[0] = sr0[0]
    sr[1:-1] = 0.5 * (sr0[-1] + sr0[1:])
    sr[-1] = sr0[-1]

    return g, sr
