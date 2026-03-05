from __future__ import annotations

import numpy as np


def chkcubair_global(dim: int, Mass_cell: np.ndarray, Mass_air: float, x: int, y: int, z: int) -> int:
    """Check if any face of a cube around (x,y,z) lies in air.

    Returns 1 if valid (not in air), 0 otherwise.
    Indices are 1-based to mirror MATLAB code.
    """

    def face(arr: np.ndarray, xs: slice, ys: slice, zs: slice) -> np.ndarray:
        return arr[xs, ys, zs]

    # convert to 0-based slices
    def sl(a: int, b: int) -> slice:
        return slice(a - 1, b)

    xface0 = face(Mass_cell, sl(x - dim, x - dim), sl(y - dim, y + dim), sl(z - dim, z + dim))
    yface0 = face(Mass_cell, sl(x - dim, x + dim), sl(y - dim, y - dim), sl(z - dim, z + dim))
    zface0 = face(Mass_cell, sl(x - dim, x + dim), sl(y - dim, y + dim), sl(z - dim, z - dim))

    xface1 = face(Mass_cell, sl(x + dim, x + dim), sl(y - dim, y + dim), sl(z - dim, z + dim))
    yface1 = face(Mass_cell, sl(x - dim, x + dim), sl(y + dim, y + dim), sl(z - dim, z + dim))
    zface1 = face(Mass_cell, sl(x - dim, x + dim), sl(y - dim, y + dim), sl(z + dim, z + dim))

    chk = 1
    if (
        np.max(xface0) <= Mass_air
        or np.max(yface0) <= Mass_air
        or np.max(zface0) <= Mass_air
        or np.max(xface1) <= Mass_air
        or np.max(yface1) <= Mass_air
        or np.max(zface1) <= Mass_air
    ):
        chk = 0

    return chk



