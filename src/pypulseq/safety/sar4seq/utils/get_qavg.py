from __future__ import annotations

import numpy as np


def _integral_volume(vol: np.ndarray) -> np.ndarray:
    """3D integral image (prefix sum) with zero padding at origin.

    Returns array of shape (M+1,N+1,P+1) where prefix[x,y,z] sums vol[:x,:y,:z].
    """
    M, N, P = vol.shape
    pref = np.zeros((M + 1, N + 1, P + 1), dtype=vol.dtype)
    pref[1:, 1:, 1:] = vol.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    return pref


def _sum_cube(pref: np.ndarray, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> float:
    """Sum over inclusive cube [x0:x1], [y0:y1], [z0:z1] using prefix with +1 padding."""
    # convert to 1-based index in prefix
    x0p, x1p = x0, x1 + 1
    y0p, y1p = y0, y1 + 1
    z0p, z1p = z0, z1 + 1
    return (
        pref[x1p, y1p, z1p]
        - pref[x0p, y1p, z1p]
        - pref[x1p, y0p, z1p]
        - pref[x1p, y1p, z0p]
        + pref[x0p, y0p, z1p]
        + pref[x0p, y1p, z0p]
        + pref[x1p, y0p, z0p]
        - pref[x0p, y0p, z0p]
    )


def get_qavg(Mass_cell: np.ndarray, Mdef: float, Qpwr2: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """Mass-averaged local Q matrices over ~10 g neighborhoods.

    Parameters
    ----------
    Mass_cell : (M,N,P) kg per voxel mass map.
    Mdef : float target mass in kg (e.g., 0.01 for 10 g).
    Qpwr2 : (M,N,P,Nc,Nc) local Q matrices per voxel.
    ms : (K,3) voxel indices (0-based) to compute outputs for.

    Returns
    -------
    Qavg_df : (M,N,P,Nc,Nc) mass-averaged Q matrices (zeros elsewhere).
    """
    M, N, P = Mass_cell.shape
    Nc = Qpwr2.shape[-1]

    # Build integral volumes
    mass_pref = _integral_volume(Mass_cell.astype(np.float64))
    massQ_pref = np.empty((Nc, Nc), dtype=object)
    for i in range(Nc):
        for j in range(Nc):
            massQ_pref[i, j] = _integral_volume((Mass_cell * Qpwr2[..., i, j]).astype(np.complex128))

    Qavg_df = np.zeros_like(Qpwr2, dtype=np.complex128)

    # Maximum reasonable radius bound
    rmax = max(M, N, P)

    for k in range(ms.shape[0]):
        x, y, z = map(int, ms[k])

        # Expand cube radius until accumulated mass >= Mdef
        r = 0
        mass_sum = 0.0
        while r <= rmax:
            x0 = max(0, x - r)
            x1 = min(M - 1, x + r)
            y0 = max(0, y - r)
            y1 = min(N - 1, y + r)
            z0 = max(0, z - r)
            z1 = min(P - 1, z + r)
            mass_sum = float(_sum_cube(mass_pref, x0, x1, y0, y1, z0, z1))
            if mass_sum >= Mdef or (x0 == 0 and x1 == M - 1 and y0 == 0 and y1 == N - 1 and z0 == 0 and z1 == P - 1):
                break
            r += 1

        if mass_sum <= 0:
            continue

        # Weighted sum of Q over cube divided by mass_sum
        for i in range(Nc):
            for j in range(Nc):
                s = _sum_cube(massQ_pref[i, j], x0, x1, y0, y1, z0, z1)
                Qavg_df[x, y, z, i, j] = s / mass_sum

    return Qavg_df



