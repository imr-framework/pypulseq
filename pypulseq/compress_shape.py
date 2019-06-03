from types import SimpleNamespace

import numpy as np


def compress_shape(decompressed_shape):
    """
    Returns a run-length encoded compressed shape.

    Parameters
    ----------
    decompressed_shape : ndarray
        Decompressed shape.

    Returns
    -------
    compressed_shape : Holder
        A Holder object containing the shape of the compressed shape ndarray and the compressed shape ndarray itself.
    """

    quant_factor = 1e-7
    decompressed_shape_scaled = decompressed_shape / quant_factor
    datq = np.round(np.insert(np.diff(decompressed_shape_scaled), 0, decompressed_shape_scaled[0]))
    qerr = decompressed_shape_scaled - np.cumsum(datq)
    qcor = np.insert(np.diff(np.round(qerr)), 0, 0)
    datd = datq + qcor
    mask_changes = np.insert(np.diff(datd) != 0, 0, 1)
    vals = np.multiply(datd[mask_changes], quant_factor).astype(np.float)

    k = np.where(np.append(mask_changes, 1) != 0)[0]
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
