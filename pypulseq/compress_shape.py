from types import SimpleNamespace

import numpy as np


def compress_shape(decompressed_shape: np.ndarray) -> np.ndarray:
    """
    Returns a run-length encoded compressed shape.

    Parameters
    ----------
    decompressed_shape : numpy.ndarray
        Decompressed shape.

    Returns
    -------
    compressed_shape : SimpleNamespace
        A `SimpleNamespace` object containing the compressed data and corresponding shape.
    """
    quant_factor = 1e-8
    decompressed_shape_scaled = decompressed_shape / quant_factor
    datq = np.round(np.insert(np.diff(decompressed_shape_scaled), 0, decompressed_shape_scaled[0]))
    qerr = decompressed_shape_scaled - np.cumsum(datq)
    qcor = np.insert(np.diff(np.round(qerr)), 0, 0)
    datd = datq + qcor
    mask_changes = np.insert(np.asarray(np.diff(datd) != 0, dtype=np.int), 0, 1)
    vals = datd[mask_changes.nonzero()[0]] * quant_factor

    k = np.append(mask_changes, 1).nonzero()[0]
    n = np.diff(k)

    n_extra = (n - 2).astype(np.float16)  # Cast as float for nan assignment to work
    vals2 = np.copy(vals)
    vals2[n_extra < 0] = np.nan
    n_extra[n_extra < 0] = np.nan
    v = np.stack((vals, vals2, n_extra))
    v = v.T[np.isfinite(v).T]  # Use transposes to match Matlab's Fortran indexing order
    v[abs(v) < 1e-10] = 0
    compressed_shape = SimpleNamespace()
    compressed_shape.num_samples = len(decompressed_shape)
    compressed_shape.data = v

    return compressed_shape
