from __future__ import annotations

import numpy as np


def gen_e12ptq(Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray, X: np.ndarray | int, SigmabyRhox: np.ndarray) -> np.ndarray:
    """Generate E^H Sigma/Rho E via 12-point cube formulation.

    Ex, Ey, Ez: 4D arrays (MxNxPxNc).
    X: 3-vector indices (x,y,z) or single linear index.
    SigmabyRhox: 3D array (MxNxP). Same used for y,z in MATLAB.
    Returns: 2D (Nc x Nc) complex matrix.
    """

    SigmabyRhoy = SigmabyRhox
    SigmabyRhoz = SigmabyRhox

    if np.isscalar(X):
        M, N, P, _ = Ex.shape
        x, y, z = np.unravel_index(int(X) - 1, (M, N, P))  # MATLAB 1-based
        x += 1; y += 1; z += 1
    else:
        x, y, z = [int(X[0]), int(X[1]), int(X[2])]

    # neighbor coordinates (MATLAB uses 1-based)
    X1 = (x, y + 1, z)
    X2 = (x, y, z + 1)
    X3 = (x, y + 1, z + 1)

    Y1 = (x + 1, y, z)
    Y2 = (x, y, z + 1)
    Y3 = (x + 1, y, z + 1)

    Z1 = (x + 1, y, z)
    Z2 = (x, y + 1, z)
    Z3 = (x + 1, y + 1, z)

    def get_E(E: np.ndarray, sigma_by_rho: float) -> np.ndarray:
        # E is (Nc,), return Nc x Nc outer product scaled
        E = E.astype(np.complex128)
        return sigma_by_rho * (E @ E.conj().T)

    def at(arr4d: np.ndarray, idx: tuple[int, int, int]) -> np.ndarray:
        # Convert 1-based to 0-based
        return arr4d[idx[0] - 1, idx[1] - 1, idx[2] - 1, :]

    def at3(arr3d: np.ndarray, idx: tuple[int, int, int]) -> float:
        return float(arr3d[idx[0] - 1, idx[1] - 1, idx[2] - 1])

    center = (x, y, z)
    Ex1 = get_E(at(Ex, center), at3(SigmabyRhox, center))
    Ey1 = get_E(at(Ey, center), at3(SigmabyRhoy, center))
    Ez1 = get_E(at(Ez, center), at3(SigmabyRhoz, center))

    Expwr = Ex1 + get_E(at(Ex, X1), at3(SigmabyRhox, X1)) + get_E(at(Ex, X2), at3(SigmabyRhox, X2)) + get_E(at(Ex, X3), at3(SigmabyRhox, X3))
    Eypwr = Ey1 + get_E(at(Ey, Y1), at3(SigmabyRhoy, Y1)) + get_E(at(Ey, Y2), at3(SigmabyRhoy, Y2)) + get_E(at(Ey, Y3), at3(SigmabyRhoy, Y3))
    Ezpwr = Ez1 + get_E(at(Ez, Z1), at3(SigmabyRhoz, Z1)) + get_E(at(Ez, Z2), at3(SigmabyRhoz, Z2)) + get_E(at(Ez, Z3), at3(SigmabyRhoz, Z3))
    Epwr = 0.125 * (Expwr + Eypwr + Ezpwr)
    return Epwr



