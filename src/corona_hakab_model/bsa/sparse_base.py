from io import BytesIO
from typing import BinaryIO, Callable

from bsa.format import BSA_Dtype, EncoderV0, decode
from sparse_base import SparseBase
from sparse_matrix import SparseMatrix


def write_sparse(matrix: SparseBase, sink: BinaryIO = None, encoder_cls=EncoderV0, **kwargs):
    if sink is None:
        sink = BytesIO()
    size = matrix.size
    nzc = matrix.non_zero_columns()
    encoder = encoder_cls(sink, size, 2, nzc, **kwargs)
    encoder.add_layer(BSA_Dtype.f32, lambda x, y: matrix[x, y][0])
    encoder.add_layer(BSA_Dtype.f32, lambda x, y: matrix[x, y][1])
    return sink


def read_sparse(source: BinaryIO, sparse_cls: Callable[[int], SparseBase] = SparseMatrix):
    decoder = decode(source)

    ret = sparse_cls(decoder.size)
    probs, vals = decoder.layers
    for layer in decoder.layers:
        if layer.dtype != BSA_Dtype.f32:
            raise TypeError("only float arrays can be accepted")
        if layer.default_value != 0:
            raise ValueError("nonzero default value")

    for row in range(ret.size):
        ret.batch_set(row, decoder.rows[row], probs.values[row], vals.values[row])

    return ret
