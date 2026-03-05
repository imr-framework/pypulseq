from __future__ import annotations

import numpy as np

from .chkcubair_global import chkcubair_global
from .gen_e12ptq import gen_e12ptq
from .get_qavg import get_qavg


def gen_qpwr(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    Tissue_types: np.ndarray,
    SigmabyRhox: np.ndarray,
    Mass_cell: np.ndarray,
    sar_type: str,
    anatomy: str,
):
    """Generate Q power matrices.

    Returns tuple analogous to MATLAB: (Qpwr_df, Tissue_types, SigmabyRhox, Mass_cell, Mass_corr, Qpwr2)
    For 'global', Qpwr2 is None.
    """

    D = Tissue_types.shape
    Mass_corr = 0.0
    Mass_air = 1.625e-7 + 1e-9

    if sar_type == 'global':
        if anatomy == 'wholebody':
            R = np.argwhere(Tissue_types > 0)
        elif anatomy == 'head':
            R = np.argwhere(Tissue_types == 1)
        elif anatomy == 'torso':
            R = np.argwhere(Tissue_types == 2)
        else:
            R = np.empty((0, 3), dtype=int)

        Qpwr = None
        Mass_corr = 0.0
        dim = 2
        for r in range(R.shape[0]):
            x, y, z = R[r]
            # to MATLAB 1-based
            xm, ym, zm = x + 1, y + 1, z + 1
            M = float(Mass_cell[x, y, z])
            chk = chkcubair_global(dim, Mass_cell, Mass_air, xm, ym, zm)
            if (Mass_air < M) and (chk == 1):
                Qloc = gen_e12ptq(Ex, Ey, Ez, np.array([xm, ym, zm]), SigmabyRhox)
                if Qpwr is None:
                    Qpwr = M * Qloc
                else:
                    Qpwr = Qpwr + (M * Qloc)
                Mass_corr += M
        Qpwr_df = Qpwr if Qpwr is not None else np.zeros((Ex.shape[3], Ex.shape[3]), dtype=np.complex128)
        return Qpwr_df, Tissue_types, SigmabyRhox, Mass_cell, Mass_corr, None

    elif sar_type == 'local':
        M, N, P = D
        ind = Mass_cell > Mass_air
        ms = np.argwhere(ind)
        Qpwr = np.zeros((ms.shape[0], 8, 8), dtype=np.complex128)
        for k in range(ms.shape[0]):
            m, n, p = ms[k]
            xm, ym, zm = m + 1, n + 1, p + 1
            Qpwr[k] = gen_e12ptq(Ex, Ey, Ez, np.array([xm, ym, zm]), SigmabyRhox)

        Qpwr2 = np.zeros((M * N * P, 8, 8), dtype=np.complex128)
        lin_inds = np.ravel_multi_index((ms[:, 0], ms[:, 1], ms[:, 2]), (M, N, P))
        Qpwr2[lin_inds] = Qpwr
        Qpwr2 = Qpwr2.reshape((M, N, P, 8, 8))

        # Average over 10 g volumes (IEC) via integral volumes
        Mdef = 0.01  # kg
        Qpwr_df = get_qavg(Mass_cell, Mdef, Qpwr2, ms)
        return Qpwr_df, Tissue_types, SigmabyRhox, Mass_cell, Mass_corr, Qpwr2

    else:
        raise ValueError("sar_type must be 'global' or 'local'")


