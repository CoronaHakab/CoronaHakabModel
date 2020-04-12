from io import BytesIO
from typing import BinaryIO, Sequence

import numpy as np
from bsa.format import BSA_Dtype, EncoderV0, decode
from scipy.sparse import lil_matrix, spmatrix


def write_scipy_sparse(matrices: Sequence[spmatrix], sink: BinaryIO = None, encoder_cls=EncoderV0, **kwargs):
    if sink is None:
        sink = BytesIO()

    size = matrices[0].shape[0]
    nz_indices = [set() for _ in range(size)]
    for m in matrices:
        if m.shape != (size, size):
            raise Exception("all matrices must be squares of the same size")
        nz_rows, nz_cols = m.nonzero()
        for i, row_set in enumerate(nz_indices):
            row_set.update(nz_cols[nz_rows == 1].tolist())
    nz_indices = [sorted(s) for s in nz_indices]
    encoder = encoder_cls(sink, size, len(matrices), nz_indices, **kwargs)
    for m in matrices:
        dtype = next(t for t in BSA_Dtype if t.np_dtype == m.dtype)
        encoder.add_layer(dtype, lambda x, y: m[x, y])
    return sink


def read_scipy_sparse(source: BinaryIO, matrix_cls=lil_matrix):
    decoder = decode(source)
    ret = []
    for layer in decoder.layers:
        if layer.default_value != 0:
            raise Exception("nonzero default_value")
        m = matrix_cls((decoder.size, decoder.size), dtype=layer.dtype.np_dtype)
        for row, (indices, vals) in enumerate(zip(decoder.rows, layer.values)):
            m[row, indices] = vals
        ret.append(m)

    return ret


if __name__ == "__main__":
    m0 = [
        np.array([[3, 0, 5, 2], [0, 0, 1, 0], [2, 4, 0, 1], [3, 0, 0, 0], ]),
        np.array([[2, 4, 0, 1], [0, 0, 1, 0], [3, 0, 0, 0], [3, 0, 5, 2], ]),
    ]

    m0 = [lil_matrix(m) for m in m0]

    b = write_scipy_sparse(m0)
    b.seek(0)

    m1 = read_scipy_sparse(b)
    for i in range(len(m0)):
        assert (m0[i] != m1[i]).nnz != 0
