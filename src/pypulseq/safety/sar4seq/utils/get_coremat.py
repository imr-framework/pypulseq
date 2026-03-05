from __future__ import annotations

import numpy as np


def get_coremat(Qinds: np.ndarray, ind: np.ndarray, myu_per: float = 0.01):
    """Implementation of steps 1 and 2 from Eichfelder VOP paper.

    Parameters
    ----------
    Qinds : (K, Nc, Nc) complex array of local Q matrices.
    ind : (K,) integer indices of observation points within a larger volume.
    myu_per : relative threshold for spectral norm defining cluster end.

    Returns
    -------
    Bstar : (Nc, Nc) complex core matrix.
    ind_sort : (K,) indices sorted by decreasing smallest eigenvalue of B*-Qk.
    vop_ind : int index of selected core matrix within original index list.
    myu_def : float absolute threshold used for clustering.
    """

    B = np.zeros(len(ind), dtype=float)
    Qind = Qinds[ind]

    for k in range(len(B)):
        Qtemp = Qind[k]
        B[k] = np.linalg.norm(Qtemp, 2)

    maxk = int(np.argmax(B))
    vop_ind = int(ind[maxk])
    myu_def = float(myu_per * B[maxk])
    Bstar = Qind[maxk]

    lambdamin = np.zeros(len(B), dtype=float)
    for k in range(len(B)):
        Qtemp = Bstar - Qind[k]
        # Use Hermitian part to ensure real eigenvalues if numerical noise
        Qh = 0.5 * (Qtemp + Qtemp.conj().T)
        vals = np.linalg.eigvalsh(Qh)
        lambdamin[k] = float(np.min(vals))

    sort_idx = np.argsort(lambdamin)[::-1]
    ind_sort = ind[sort_idx]

    return Bstar, ind_sort, vop_ind, myu_def



