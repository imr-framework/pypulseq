from __future__ import annotations

import numpy as np


def read_qmat(fname: str):
    fname1 = f'{fname}.qmat'
    with open(fname1, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=3)

    with open(fname1, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    data = data[4:]

    n_coils = int(header[2 - 1])  # header[1] in MATLAB
    n_cells = int(header[3 - 1])

    if n_cells == 3:
        R = data[0::2]
        I = data[1::2]
        Qavg = R + 1j * I
        third = Qavg.size // 3
        Qtm = Qavg[:third]
        Qhm = Qavg[third : 2 * third]
        Qem = Qavg[2 * third :]

        def upper_to_full(vec):
            mat = np.triu(np.ones((n_coils, n_coils), dtype=np.complex128))
            mat[mat == 1] = vec.astype(np.complex128)
            return np.triu(mat, 1).T + mat

        Qtmf = upper_to_full(Qtm)
        Qhmf = upper_to_full(Qhm)
        Qemf = upper_to_full(Qem)
        return {'Qtmf': Qtmf, 'Qhmf': Qhmf, 'Qemf': Qemf}

    # Local matrices case
    index_name = f'{fname}.index'
    with open(index_name, 'rb') as f:
        data_ind = np.fromfile(f, dtype=np.uint32, count=4)
    dimx, dimy, dimz, n_sar_cells = map(int, data_ind)
    with open(index_name, 'rb') as f:
        data_ind16 = np.fromfile(f, dtype=np.uint16)

    index = np.zeros((4, n_sar_cells), dtype=np.uint16)
    index[0, :] = data_ind16[8 + 1 :: 4]  # x
    index[1, :] = data_ind16[8 + 2 :: 4]  # y
    index[2, :] = data_ind16[8 + 3 :: 4]  # z
    index[3, :] = data_ind16[8 + 4 :: 4]  # label

    R = data[0::2]
    I = data[1::2]
    Qstore = (R + 1j * I).reshape(36, n_sar_cells)
    Qavg = np.zeros((n_sar_cells, n_coils, n_coils), dtype=np.complex128)

    # Fill upper triangular then symmetrize
    for k in range(n_sar_cells):
        mat = np.triu(np.ones((n_coils, n_coils), dtype=np.complex128))
        mat[mat == 1] = Qstore[:, k].astype(np.complex128)
        Qavg[k] = np.triu(mat, 1).T + mat

    return {
        'avg': Qavg,
        'index': index,
        'dimx': dimx,
        'dimy': dimy,
        'dimz': dimz,
        'NrOfSarCells': n_sar_cells,
    }



