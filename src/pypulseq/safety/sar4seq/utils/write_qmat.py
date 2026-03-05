from __future__ import annotations

import numpy as np


def _write_uppertri(Qin: np.ndarray, f) -> None:
    Q = np.triu(Qin)
    Q_tri = Q[np.abs(Q) > 0]
    for val in Q_tri:
        np.array([np.real(val)], dtype=np.float32).tofile(f)
        np.array([np.imag(val)], dtype=np.float32).tofile(f)


def write_qmat_global(Q: dict, out_path: str) -> None:
    with open(out_path, "wb") as f:
        n_coils = int(Q["Qtmf"].shape[0])
        n_cells = 3
        np.array([n_coils], dtype=np.uint32).tofile(f)
        np.array([n_coils], dtype=np.uint32).tofile(f)
        np.array([n_cells], dtype=np.uint32).tofile(f)

        _write_uppertri(Q["Qtmf"], f)
        _write_uppertri(Q["Qhmf"], f)
        # Optional: Qemf


def write_qmat_local(Qavg: np.ndarray, Tissue_types: np.ndarray, tri_path: str, index_path: str) -> None:
    M, N, P, _, n_coils = Qavg.shape
    flat = Qavg.reshape(M * N * P, n_coils, n_coils)
    S = flat > 0
    ind = np.where(S[:, 3, 3])[0]
    n_cells = int(len(ind))

    with open(tri_path, "wb") as tri_f, open(index_path, "wb") as idx_f:
        np.array([n_coils], dtype=np.uint32).tofile(tri_f)
        np.array([n_coils], dtype=np.uint32).tofile(tri_f)
        np.array([n_cells], dtype=np.uint32).tofile(tri_f)

        np.array([M], dtype=np.uint32).tofile(idx_f)
        np.array([N], dtype=np.uint32).tofile(idx_f)
        np.array([P], dtype=np.uint32).tofile(idx_f)
        np.array([n_cells], dtype=np.uint32).tofile(idx_f)

        for k in ind:
            x, y, z = np.unravel_index(k, (M, N, P))
            label = int(Tissue_types[x, y, z])
            _write_uppertri(flat[k], tri_f)
            np.array([x], dtype=np.uint16).tofile(idx_f)
            np.array([y], dtype=np.uint16).tofile(idx_f)
            np.array([z], dtype=np.uint16).tofile(idx_f)
            np.array([label], dtype=np.uint16).tofile(idx_f)



