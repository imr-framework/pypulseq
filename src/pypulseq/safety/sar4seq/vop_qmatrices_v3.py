from __future__ import annotations

import numpy as np

from .utils.get_coremat import get_coremat


def vop_qmatrices(Qavg_imp: np.ndarray):
    """Port of VOP_Qmatrices_v3.m without GUI. Accepts Qavg.imp array.

    Parameters
    ----------
    Qavg_imp : ndarray
        Array shaped (M, N, P, 8, 8) of local Q matrices.

    Returns
    -------
    VOP_imp : ndarray
        Sparse VOP matrices placed back into volume (M, N, P, 8, 8).
    normplot : ndarray
        Spectral norms of selected VOPs.
    vop_ind : ndarray
        Linear indices of VOP locations within volume.
    """

    M, N, P, _, _ = Qavg_imp.shape
    Qavg_df = Qavg_imp.reshape(M * N * P, 8, 8)

    S = np.abs(Qavg_df) > 0
    ind = np.where(S[:, 3, 3])[0]
    Qinds = Qavg_df[ind]
    Qind = Qinds.copy()

    obs_pts = len(ind)
    cluster = np.zeros(100, dtype=int)
    normplot = np.zeros(100, dtype=float)
    vop_ind = np.zeros(100, dtype=int)
    VOPm = np.zeros((500, 8, 8), dtype=np.complex128)
    VOP = 0

    indr = np.arange(1, len(ind) + 1)

    while obs_pts != 0:
        lenindr = len(indr)
        if lenindr == len(ind):
            Bstar, ind_sorta, vopin, myu_def = get_coremat(Qind, indr - 1)
        elif lenindr == 1:
            break
        else:
            Bstar, ind_sorta, vopin, _ = get_coremat(Qind, indr - 1)

        q = 2
        A = Bstar.copy()
        Z = np.zeros((8, 8), dtype=np.complex128)
        cluster_done = False
        obs_pts -= 1
        while not cluster_done:
            Qm = A - Qind[ind_sorta[q - 1]]
            # Spectral decomposition
            vals, V = np.linalg.eigh(Qm)
            Ep = np.diag(np.maximum(vals, 0))
            Em = Ep - np.diag(vals)
            Z_new = V @ Em @ V.conj().T
            Z = Z + Z_new
            myu_calc = float(np.linalg.norm(Z, 2))

            if myu_calc >= myu_def:
                cluster_done = True
                VOP += 1
                VOPm[VOP - 1] = A
                cluster[VOP - 1] = q - 1
                normplot[VOP - 1] = np.linalg.norm(Bstar)
                # remove used indices
                used = ind_sorta[: q - 1]
                mask = np.ones_like(indr, dtype=bool)
                mask[used] = False
                indr = indr[mask]
                vop_ind[VOP - 1] = ind[vopin]
            else:
                if q < len(ind_sorta):
                    A = Bstar + Z
                    obs_pts -= 1
                    q += 1
                elif q == len(ind_sorta):
                    obs_pts -= 1
                    cluster_done = True
                    VOP += 1
                    VOPm[VOP - 1] = A
                    cluster[VOP - 1] = q
                    normplot[VOP - 1] = np.linalg.norm(Bstar)
                    vop_ind[VOP - 1] = ind[vopin]
                    used = ind_sorta[:q]
                    mask = np.ones_like(indr, dtype=bool)
                    mask[used] = False
                    indr = indr[mask]
                    break

    VOPm = VOPm[:VOP]
    normplot = normplot[:VOP]
    vop_ind = vop_ind[:VOP]

    VOP_imp = np.zeros((M, N, P, 8, 8), dtype=np.complex128)
    for k in range(VOP):
        x, y, z = np.unravel_index(int(vop_ind[k]), (M, N, P))
        VOP_imp[x, y, z] = VOPm[k]

    return VOP_imp, normplot, vop_ind



