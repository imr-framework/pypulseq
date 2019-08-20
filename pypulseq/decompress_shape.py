import numpy as np


def decompress_shape(compressed_shape: np.ndarray) -> np.ndarray:
    """
    Decompresses a run-length encoded shape.

    Parameters
    ----------
    compressed_shape : numpy.ndarray
        Run-length encoded shape.

    Returns
    -------
    decompressed_shape : numpy.ndarray
        Decompressed shape.
    """
    data_pack, num_samples = compressed_shape.data, int(compressed_shape.num_samples)
    decompressed_shape = np.zeros(num_samples)

    count_pack, count_unpack = 0, 0
    while count_pack < max(data_pack.shape) - 1:
        if data_pack[count_pack] != data_pack[count_pack + 1]:
            decompressed_shape[count_unpack] = data_pack[count_pack]
            count_unpack += 1
            count_pack += 1
        else:
            rep = int(data_pack[count_pack + 2] + 2)
            decompressed_shape[count_unpack:(count_unpack + rep)] = data_pack[count_pack]
            count_pack += 3
            count_unpack += rep

    if count_pack == max(data_pack.shape) - 1:
        decompressed_shape[count_unpack] = data_pack[count_pack]

    decompressed_shape = np.cumsum(decompressed_shape)
    return decompressed_shape
