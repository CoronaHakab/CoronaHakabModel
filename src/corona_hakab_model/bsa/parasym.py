from functools import partial
from io import BytesIO
from itertools import chain, count
from typing import BinaryIO

from bsa.format import BSA_Dtype, EncoderV0, decode
from parasymbolic_matrix import ParasymbolicMatrix


def write_parasym(matrix: ParasymbolicMatrix, sink: BinaryIO = None, encoder_cls=EncoderV0, **kwargs):
    if sink is None:
        sink = BytesIO()
    size = matrix.get_size()
    nzc = matrix.non_zero_columns()
    depth = len(nzc)
    comb_nzc = [sorted(set(chain.from_iterable(comp[r] for comp in nzc))) for r in range(size)]
    del nzc
    encoder = encoder_cls(sink, size, depth, comb_nzc, **kwargs)
    for d in range(depth):
        encoder.add_layer(BSA_Dtype.f32, partial(matrix.get, d))

    return sink


def read_parasym(source: BinaryIO):
    decoder = decode(source)
    ret = ParasymbolicMatrix(decoder.size, decoder.layer_count)
    with ret.lock_rebuild():
        for d, layer in enumerate(decoder.layers):
            if layer.default_value != 0:
                raise ValueError("nonzero default value is not supported")
            if layer.dtype != BSA_Dtype.f32:
                raise ValueError("parasymbolic can only accept float32 layers")
            for i, indices, values in zip(count(), decoder.rows, layer.values):
                ret[d, i, indices] = values
    return ret
