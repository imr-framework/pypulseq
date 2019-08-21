import math

import numpy as np

from pypulseq.opts import Opts
from pypulseq.utils import k2g


def vds_2d(fov: int, N: int, n_shots: int, alpha: int, system: Opts):
    """
    Generates a variable density k-space trajectory spiral with a method adapted from [1].

    [1] "Simple Analytic Variable Density Spiral Design", Dong-hyun Kim, Elfar Adalsteinsson, and Daniel M. Spielman,
    Magnetic Resonance in Medicine 50:214-219 (2003).

    Parameters
    ----------
    fov : int
        Field of view in meters.
    N : int
        Resolution (Nyquist distance) in meters.
    n_shots : int
        Number of interleaves.
    alpha : int
        Variable density factor.
    system : Opts
        System limits.

    Returns
    -------
    k_shot : numpy.ndarray
        K-space trajectory for the nth shot.
    Gn : numpy.ndarray
        Gradient waveform for the nth shot.
    lamda : float
        Inter-shot distance factor.
    """

    gamma = 42576000
    max_grad = system.max_grad / gamma
    slew_safety = 0.8
    max_slew = system.max_slew / gamma * slew_safety
    res = fov / N
    lamda = 0.5 / res
    n = (1 / (1 - pow(1 - n_shots / fov / lamda, 1 / alpha)))
    w = 2 * math.pi * n
    tea = lamda * w / gamma / max_grad / (alpha + 1)
    tes = math.sqrt(lamda * pow(w, 2) / max_slew / gamma) / (alpha / 2 + 1)
    ts2a = pow((pow(tes, (alpha + 1) / (alpha / 2 + 1)) * (alpha / 2 + 1) / tea / (alpha + 1)), (1 + 2 / alpha))

    if ts2a < tes:
        tau_trans = pow(ts2a / tes, 1 / (alpha / 2 + 1))
        tau = lambda t: pow(t / tes, 1 / (alpha / 2 + 1)) * (0 <= t) * (t <= ts2a) + pow(
            (((t - ts2a) / tea) + pow(tau_trans, alpha + 1)), 1 / (alpha + 1)) * (t > ts2a) * (t <= tea) * (tes >= ts2a)
        t_end = tea
    else:
        tau = lambda t: pow((t / tes), (1 / (alpha / 2 + 1)) * (0 <= t) * (t <= tes))
        t_end = tes

    k = lambda t: lamda * tau(t) ** alpha * pow(math.e, 1j * w * tau(t))
    dt = tea * 1e-4
    Dt = dt / fov / abs(k(tea) - k(tea - dt))
    t = np.arange(0, t_end, Dt)
    kt = k(t)

    DT = system.grad_raster_time
    tn = np.arange(0, t_end, DT)
    kn = k(tn)

    ktn = np.array([np.real(kn), np.imag(kn)])
    Gx, Gy, _ = k2g.k2g(ktn, DT)

    k_shot = np.zeros((len(kn), n_shots)).astype(np.complex)
    Gn = np.zeros((len(kn), n_shots)).astype(np.complex)
    complex_vectorized = np.vectorize(lambda x, y: complex(x, y))
    for s in range(n_shots):
        k_shot[:, s] = kn * pow(math.e, 2 * math.pi * 1j * (s + 1) / n_shots)
        km = np.array([np.real(np.squeeze(k_shot[:, s])), np.imag(np.squeeze(k_shot[:, s]))])
        Gx, Gy, _ = k2g.k2g(km, DT)
        Gn[:, s] = complex_vectorized(Gx, Gy)

    return k_shot, Gn, lamda
