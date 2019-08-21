import numpy as np


def k2g(k: np.ndarray, dt: float = 10e-6):
    """
    Derives gradient-waveforms from k-space trajectory in compliance with the dwell time.

    Parameters
    ----------
    k : numpy.ndarray
        Target k-space trajectory.
    dt : float
        Dwell time in milliseconds (ms). Default is 10e-6.
    Returns
    -------
    Gx, Gy, Gz  :numpy.ndarray
        Gradient waveforms to accomplish `k`.
    """
    factor = 1
    Gx = (factor / dt) * (np.diff(np.squeeze(k[0, :])))
    Gy = (factor / dt) * (np.diff(np.squeeze(k[1, :])))
    Gz = (factor / dt) * (np.diff(np.squeeze(k[2, :]))) if np.size(k, 0) > 2 else 0
    Gx = np.insert(Gx, 0, 0)
    Gy = np.insert(Gy, 0, 0)
    Gz = np.insert(Gz, 0, 0)
    return Gx, Gy, Gz
