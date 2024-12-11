from types import SimpleNamespace

import numpy as np


def decompress_shape(compressed_shape: SimpleNamespace, force_decompression: bool = False) -> np.ndarray:
    """
    Decompress a gradient or pulse shape compressed with a run-length compression scheme on the derivative. The given
    shape is structure with the following fields:
    - num_samples - the number of samples in the uncompressed waveform
    - data - containing the compressed waveform

    See also `compress_shape.py`.

    Parameters
    ----------
    compressed_shape : SimpleNamespace
        Run-length encoded shape.
    force_decompression : bool, default=False

    Returns
    -------
    decompressed_shape : numpy.ndarray
        Decompressed shape.
    """
    data_pack = compressed_shape.data
    data_pack_len = len(data_pack)
    num_samples = int(compressed_shape.num_samples)

    if not force_decompression and num_samples == data_pack_len:
        # Uncompressed shape
        decompressed_shape = data_pack
        return decompressed_shape

    decompressed_shape = np.zeros(num_samples)  # Pre-allocate result matrix

    # Decompression starts here
    data_pack_diff = data_pack[1:] - data_pack[:-1]

    # When data_pack_diff == 0 the subsequent samples are equal ==> marker for repeats (run-length encoding)
    data_pack_markers = np.where(data_pack_diff == 0.0)[0]

    count_pack = 0  # Points to current compressed sample
    count_unpack = 0  # Points to current uncompressed sample

    for i in range(len(data_pack_markers)):
        # This index may have "false positives", e.g. if the value 3 repeats 3 times, then we will have 3 3 3
        next_pack = data_pack_markers[i]
        current_unpack_samples = next_pack - count_pack
        if current_unpack_samples < 0:  # Rejects false positives
            continue
        elif current_unpack_samples > 0:  # We have an unpacked block to copy
            decompressed_shape[count_unpack : count_unpack + current_unpack_samples] = data_pack[count_pack:next_pack]
            count_pack += current_unpack_samples
            count_unpack += current_unpack_samples

        # Packed/repeated section
        rep = int(data_pack[count_pack + 2] + 2)
        decompressed_shape[count_unpack : (count_unpack + rep)] = data_pack[count_pack]
        count_pack += 3
        count_unpack += rep

    # Samples left?
    if count_pack <= data_pack_len - 1:
        assert data_pack_len - count_pack == num_samples - count_unpack
        # Copy the rest of the shape, it is unpacked
        decompressed_shape[count_unpack:] = data_pack[count_pack:]

    decompressed_shape = np.cumsum(decompressed_shape)
    return decompressed_shape
