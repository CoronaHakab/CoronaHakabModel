from io import BytesIO
from typing import BinaryIO, Sequence

import numpy as np
from bsa.format import BSA_Dtype, EncoderV0, decode


def write_ndarr(matrices: Sequence[np.ndarray], sink: BinaryIO = None, encoder_cls=EncoderV0, **kwargs):
    if sink is None:
        sink = BytesIO()

    size = matrices[0].shape[0]
    nz_indices = [set() for _ in range(size)]
    for m in matrices:
        if m.shape != (size, size):
            raise Exception("all matrices must be squares of the same size")
        for row_set, row_m in zip(nz_indices, m):
            row_set.update(np.flatnonzero(row_m).tolist())
    nz_indices = [sorted(s) for s in nz_indices]
    encoder = encoder_cls(sink, size, len(matrices), nz_indices, **kwargs)
    for m in matrices:
        dtype = next(t for t in BSA_Dtype if t.np_dtype == m.dtype)
        encoder.add_layer(dtype, m.item)
    return sink


def read_ndarr(source: BinaryIO):
    decoder = decode(source)
    ret = []
    for layer in decoder.layers:
        m = np.full((decoder.size, decoder.size), layer.default_value, layer.dtype.np_dtype)
        for row, (indices, vals) in enumerate(zip(decoder.rows, layer.values)):
            m[row, indices] = vals
        ret.append(m)

    return ret


if __name__ == "__main__":
    m0 = [
        np.array([[3, 0, 5, 2], [0, 0, 1, 0], [2, 4, 0, 1], [3, 0, 0, 0], ]),
        np.array([[2, 4, 0, 1], [0, 0, 1, 0], [3, 0, 0, 0], [3, 0, 5, 2], ]),
    ]

    b = write_ndarr(m0)
    b.seek(0)

    m1 = read_ndarr(b)
    for i in range(len(m0)):
        assert np.allclose(m0[i], m1[i])
