from __future__ import annotations

from array import array
from enum import IntFlag, IntEnum
from itertools import chain
from typing import BinaryIO, Sequence, Collection, Iterable, Optional, Callable, Any

import numpy as np


class Flags(IntFlag):
    pass


class BSA_Dtype(IntEnum):
    def __new__(cls, code, length, array_format, np_dtype: np.dtype):
        self = int.__new__(cls, code)
        self._value_ = code

        self.length = length
        self.array_format = array_format
        self.np_dtype = np_dtype
        return self

    i8 = 1, 1, 'b', np.int8
    i16 = 2, 2, 'h', np.int16
    i32 = 3, 4, 'l', np.int32
    i64 = 4, 8, 'q', np.int64

    u8 = 5, 1, 'B', np.uint8
    u16 = 6, 2, 'H', np.uint16
    u32 = 7, 4, 'L', np.uint32
    u64 = 8, 8, 'Q', np.uint64

    f32 = 9, 4, 'f', np.float32
    f64 = 10, 8, 'd', np.float64

    a8 = 251, 1, 'B', np.uint8
    a16 = 252, 2, 'H', np.uint16
    a32 = 253, 4, 'L', np.uint32
    a64 = 254, 8, 'Q', np.uint64

    def value_to_bytes(self, v) -> bytes:
        return array(self.array_format, [v]).tobytes()

    def bytes_to_value(self, b: bytes):
        arr = array(self.array_format)
        arr.frombytes(b)
        return arr[0]


class EncoderV0:
    magic = b'SA'
    version = 0

    def __init__(self, sink: BinaryIO, size: int, layer_count: int, indices: Sequence[Collection[int]],
                 user_data: bytes = b'', flags: Flags = Flags(0)):
        self.sink = sink

        # header
        self._write(self.magic)
        self._write_uint(self.version, 1)
        self._write_with_len(user_data, 2)
        self._write_uint(flags, 2)
        self._write_uint(size, 4)
        self._write_uint(layer_count, 1)

        self.indices = indices
        # rows
        self._write_uints(
            (len(r) for r in self.indices), 4
        )
        # columns
        self._write_uints(chain.from_iterable(self.indices), 4)

    def add_layer(self, dtype: BSA_Dtype, map_: Callable[[int, int], Any], user_data: bytes = b'', default=0):
        self._write_uint(int(dtype), 1)
        self._write_with_len(user_data, 2)
        self._write(dtype.value_to_bytes(default))
        arr = array(dtype.array_format)
        for i, row in enumerate(self.indices):
            arr.extend(
                map_(i, j) for j in row
            )
        self._write(arr.tobytes())

    def _write_many(self, b: Iterable[bytes]):
        self.sink.writelines(b)

    def _write(self, b: bytes):
        self.sink.write(b)

    def _write_uint(self, i: int, chars: int):
        self._write(i.to_bytes(chars, 'little'))

    def _write_uints(self, i: Iterable[int], chars: int):
        # todo improve?
        self._write_many((
            n.to_bytes(chars, 'little')
            for n in i
        ))

    def _write_with_len(self, b: bytes, lenlen: int):
        self._write_uint(len(b), lenlen)
        self._write(b)


class DecoderV0:
    class Layer:
        def __init__(self, owner: DecoderV0, layer_ind: int):
            self.dtype = BSA_Dtype(owner._read_uint(1))
            self.ud = owner._read_with_len(2)
            owner.handle_ud(self.ud, layer_ind)
            self.default_value = self.dtype.bytes_to_value(owner._read(self.dtype.length))

            buffer = owner._read(owner.layer_size * self.dtype.length)
            arr = array(self.dtype.array_format)
            arr.frombytes(buffer)

            self.values = []
            i = 0
            for row in owner.rows:
                self.values.append(
                    arr[i:i + len(row)]
                )
                i += len(row)

    def __init__(self, source: BinaryIO, read_magic: bool):
        self.source = source

        if read_magic:
            self._read(2)  # magic
            self._read_uint(1)  # version

        self.ud = self._read_with_len(2)
        self.handle_ud(self.ud, None)
        self.flags = Flags(self._read_uint(2))
        self.size = self._read_uint(4)
        self.layer_count = self._read_uint(1)

        row_lengths = list(self._read_uints(4, self.size))
        self.layer_size = sum(row_lengths)
        cols_flat = list(self._read_uints(4, self.layer_size))
        i = 0
        self.rows = []
        for row_len in row_lengths:
            self.rows.append(
                cols_flat[i: i + row_len]
            )
            i += row_len
        assert i == self.layer_size

        self.layers = []
        for i in range(self.layer_count):
            self.layers.append(
                self.Layer(self, i)
            )

    def handle_ud(self, data: bytes, layer: Optional[int]):
        if data:
            raise Exception("cannot handle user data")

    def _read(self, n: int):
        return self.source.read(n)

    def _read_uint(self, n: int):
        return int.from_bytes(self._read(n), 'little')

    def _read_uints(self, n: int, count: int):
        # todo improve?
        yield from (
            self._read_uint(n) for _ in range(count)
        )

    def _read_with_len(self, lenlen: int):
        n = self._read_uint(lenlen)
        return self._read(n)


decoders = {
    (b'SA', 0): DecoderV0
}


def decode(source: BinaryIO) -> DecoderV0:
    magic = source.read(2)
    version = source.read(1)[0]
    return decoders[magic, version](source, False)
