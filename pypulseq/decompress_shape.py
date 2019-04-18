import numpy as np

from pypulseq.holder import Holder


def decompress_shape(compressed_shape):
    """
    Decompresses a run-length encoded shape.

    Parameters
    ----------
    compressed_shape : ndarray
        Run-length encoded shape.

    Returns
    -------
    decompressed_shape : ndarray
        Decompressed shape.
    """
    if compressed_shape.data.shape[0] != 1:
        raise ValueError("input should be of shape (1,x)")
    if not isinstance(compressed_shape, Holder):
        raise TypeError("input should be of type holder.Holder")

    data_pack, num_samples = compressed_shape.data, int(compressed_shape.num_samples)
    decompressed_shape = np.zeros(num_samples)

    count_pack, count_unpack = 0, 0
    while count_pack < max(data_pack.shape) - 1:

        if data_pack[0][count_pack] != data_pack[0][count_pack + 1]:
            decompressed_shape[count_unpack] = data_pack[0][count_pack]
            count_unpack += 1
            count_pack += 1
        else:
            rep = int(data_pack[0][count_pack + 2] + 2)
            decompressed_shape[count_unpack:(count_unpack + rep - 1)] = data_pack[0][count_pack]
            count_pack += 3
            count_unpack += rep

    if count_pack == max(data_pack.shape) - 1:
        decompressed_shape[count_unpack] = data_pack[0][count_pack]

    decompressed_shape = np.cumsum(decompressed_shape)
    return decompressed_shape
