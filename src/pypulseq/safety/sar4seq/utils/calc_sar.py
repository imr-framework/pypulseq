from __future__ import annotations

import numpy as np


def calc_sar(Q: np.ndarray, I: np.ndarray, weight: float) -> float:
    """Compute SAR given a Q matrix and RF waveform.

    Parameters
    ----------
    Q : np.ndarray
        Either a 2D complex matrix (Nc x Nc) or 3D (K x Nc x Nc) of Q matrices.
    I : np.ndarray
        RF waveform array shaped (Nc, Nt) or (Nc,) complex.
    weight : float
        Mass in kg used for normalization when Q is 2D.

    Returns
    -------
    float
        SAR value.
    """

    I = np.asarray(I)
    if I.ndim == 1:
        I = I[:, None]

    # Iexp = mean over time of |I|^2 assuming 1 channel unless provided
    Iexp = np.conj(I) * I
    Iexp = np.sum(Iexp) / Iexp.size
    Ifact = Iexp

    Q = np.asarray(Q)
    if Q.ndim > 2:
        sar_temp = np.zeros_like(Q, dtype=np.complex128)
        sar_norm = np.zeros(Q.shape[0], dtype=float)
        for k in range(Q.shape[0]):
            Qtemp = Q[k]
            sar_temp[k] = Qtemp * Ifact
            sar_norm[k] = np.linalg.norm(sar_temp[k])
        ind = int(np.argmax(sar_norm))
        sar_chosen = sar_temp[ind]
        sar = np.abs(np.sum(sar_chosen))
    else:
        sar_temp = Q * Ifact
        sar = np.abs(np.sum(sar_temp))
        sar = sar / float(weight)

    return float(np.real(sar))



