from types import SimpleNamespace

import numpy as np


def compress_shape(
    decompressed_shape: np.ndarray, force_compression: bool = False
) -> SimpleNamespace:
    """
    Compress a gradient or pulse shape waveform using a run-length compression scheme on the derivative. This strategy
    encodes constant and linear waveforms with very few samples. A structure is returned with the fields:
    - num_samples - the number of samples in the uncompressed waveform
    - data - containing the compressed waveform

    See also `pypulseq.decompress_shape.py`.

    Parameters
    ----------
    decompressed_shape : numpy.ndarray
        Decompressed shape.
    force_compression: bool, default=False
        Boolean flag to indicate if compression is forced.

    Returns
    -------
    compressed_shape : SimpleNamespace
        A `SimpleNamespace` object containing the number of samples and the compressed data.
    """
    if np.any(~np.isfinite(decompressed_shape)):
        raise ValueError("compress_shape() received infinite samples.")

    if (
        not force_compression and len(decompressed_shape) <= 4
    ):  # Avoid compressing very short shapes
        compressed_shape = SimpleNamespace()
        compressed_shape.num_samples = len(decompressed_shape)
        compressed_shape.data = decompressed_shape
        return compressed_shape

    # Single precision floating point has ~7.25 decimal places
    quant_factor = 1e-7
    decompressed_shape_scaled = decompressed_shape / quant_factor
    datq = np.round(
        np.concatenate((decompressed_shape_scaled[[0]], np.diff(decompressed_shape_scaled)))
    )
    qerr = decompressed_shape_scaled - np.cumsum(datq)
    qcor = np.concatenate(([0], np.diff(np.round(qerr))))
    datd = datq + qcor

    # RLE of datd
    starts = np.concatenate(([0], np.flatnonzero(datd[1:] != datd[:-1])+1))
    lengths = np.diff(np.concatenate((starts, [len(datd)])))
    values = datd[starts] * quant_factor
    
    # Repeat values of any run-length>1 three times: (value, value, length)
    rl_gt1 = lengths>1
    repeats = 1 + rl_gt1*2
    v = np.repeat(values, repeats)
    
    # Calculate indices of length elements and insert length values
    inds = np.cumsum(repeats) - 1
    v[inds[rl_gt1]] = lengths[rl_gt1] - 2

    compressed_shape = SimpleNamespace()
    compressed_shape.num_samples = len(decompressed_shape)

    # Decide whether compression makes sense, otherwise store the original
    if force_compression or compressed_shape.num_samples > len(v):
        compressed_shape.data = v
    else:
        compressed_shape.data = decompressed_shape

    return compressed_shape
