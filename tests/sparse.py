from io import BytesIO
from itertools import product
from typing import Union

import numpy as np

from bsa.sparse_base import write_sparse, read_sparse
from clustered_matrix import ClusteredSparseMatrix
from consts import generator
from sparse_matrix import MagicOperator, SparseMatrix


class npSparseMatrix:
    def __init__(self, size):
        self.size = size
        self.probs = np.zeros((size, size), dtype=np.float32)
        self.vals = np.zeros((size, size), dtype=np.float32)

        self.prob_coff_row = np.ones(size, dtype=np.float32)
        self.prob_coff_col = np.ones(size, dtype=np.float32)

        self.val_offs_row = np.zeros(size, dtype=np.float32)
        self.val_offs_col = np.zeros(size, dtype=np.float32)

    def batch_set(self, row, cols, probs, vals):
        self.probs[row][cols] = probs
        self.vals[row][cols] = vals

    def has_value(self, i, j):
        return self.probs[i, j] != 0

    def get(self, i, j):
        return self.probs[i, j], self.vals[i, j]

    def __getitem__(self, item):
        i, j = item
        if not self.has_value(i, j):
            return None
        return self.get(i, j)

    def nz_count(self):
        return np.count_nonzero(self.probs)

    def row_set_prob_coff(self, row, coff):
        self.prob_coff_row[row] = coff

    def col_set_prob_coff(self, col, coff):
        self.prob_coff_col[col] = coff

    def row_set_value_offset(self, row, offs):
        self.val_offs_row[row] = offs

    def col_set_value_offset(self, col, offs):
        self.val_offs_col[col] = offs

    def probs_actual(self):
        ret = np.copy(self.probs)
        for i in range(self.size):
            ret[i] *= self.prob_coff_row[i]
            ret[:, i] *= self.prob_coff_col[i]
        return ret

    def manifest(self, sample=None):
        pactual = self.probs_actual()
        if sample is None:
            sample = generator.random(self.nz_count())
        s = iter(sample)
        inner = np.zeros_like(self.vals)
        for row, col in product(range(self.size), repeat=2):
            p = pactual[row, col]
            if self.probs[row, col] != 0:
                r = next(s)
                if r < p:
                    inner[row, col] = self.vals[row, col] + self.val_offs_row[row] + self.val_offs_col[col]

        return npManifest(self, inner)

    def non_zero_columns(self):
        return [
            list(np.flatnonzero(self.probs[i])) for i in range(self.size)
        ]


class npManifest:
    def __init__(self, origin, inner):
        self.origin = origin
        self.inner = inner

    def __getitem__(self, item):
        return self.inner[item]

    def I_POA(self, v, op):
        ret = np.ones(self.origin.size, dtype=np.float32)
        for row, col in product(range(self.origin.size), repeat=2):
            ret[row] *= op.operate(self.inner[row, col], v[col])
        return ret

    def nz_rows(self):
        ret = []
        for row in self.inner:
            ret.append(list(np.flatnonzero(row)))
        return ret


v = np.array([0.2, 0.3, 0.6, 0], dtype=np.float32)
magic = MagicOperator()


def check_equal(ps: SparseMatrix, mck: npSparseMatrix, msg: str):
    for i, j in product(range(mck.size), repeat=2):
        p = ps[i, j]
        m = mck[i, j]
        assert (p is m) or np.allclose(p, m), f"{msg}, [{i},{j}] {p} vs {m}"
    assert ps.nz_count() == mck.nz_count(), f"{msg}, nz"
    sample = generator.random(ps.nz_count(), dtype=np.float32)
    m = ps.manifest(sample)
    mm = mck.manifest(sample)
    for i, j in product(range(mck.size), repeat=2):
        p_v = m[i, j]
        m_v = mm[i, j]
        assert np.allclose(p_v, m_v), f"{msg}, manifest, [{i},{j}] {p_v} vs {m_v}"
    i = m.I_POA(v, magic)
    im = mm.I_POA(v, magic)
    assert np.allclose(i, im), f"{msg}, IOP {i} vs {im}, inner: \n{mm.inner}"
    mnz = m.nz_rows()
    mmnz = mm.nz_rows()
    assert mnz == mmnz, f"{msg}, NZR {mnz} vs {mmnz}, inner: \n{mm.inner}"
    cnz = ps.non_zero_columns()
    mcnz = mck.non_zero_columns()
    assert cnz == mcnz, f"{msg}, NZC {cnz} vs {mcnz}"


def operate(matrix: Union[npSparseMatrix, SparseMatrix]):
    yield "pre_set"
    matrix.batch_set(0, [1, 2, 3], [0.2, 0.5, 1], [0.1, 0.1, 0.2])
    matrix.batch_set(1, [0, 3], [0.1, 0.1], [0.7, 1])
    matrix.batch_set(2, [0], [0.7], [0.5])
    matrix.batch_set(3, [2], [0.1], [0.9])
    yield "post_set"
    matrix.row_set_prob_coff(2, 0.5)
    yield "mr"
    matrix.col_set_prob_coff(0, 0)
    yield "mc"
    matrix.row_set_value_offset(2, 1)
    yield "or"
    matrix.col_set_value_offset(1, 1)
    yield "oc"


def test_sparse():
    main = SparseMatrix(4)
    mock = npSparseMatrix(4)

    m = operate(main)
    c = operate(mock)
    for i, j in zip(m, c):
        if __name__ == "__main__":
            print(i)
        assert i == j
        check_equal(main, mock, i)


def test_read_write():
    matrix = SparseMatrix(4)
    matrix.batch_set(0, [1, 2, 3], [0.2, 0.5, 1], [0.1, 0.1, 0.2])
    matrix.batch_set(1, [0, 3], [0.1, 0.1], [0.7, 1])
    matrix.batch_set(2, [0], [0.7], [0.5])
    matrix.batch_set(3, [2], [0.1], [0.9])

    sink = BytesIO()
    write_sparse(matrix, sink)
    sink.seek(0)
    dec = read_sparse(sink)

    for i, j in product(range(4), repeat=2):
        assert matrix[i, j] == dec[i, j], f'[{i}, {j}], {matrix[i, j]} vs {dec[i, j]}'

def test_csm_constructible():
    # this is just a test to check that CSM is non-abstract
    csm = ClusteredSparseMatrix(
        [
            [0,1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    )

if __name__ == "__main__":
    matrix = SparseMatrix(4)
    matrix.manifest()
    test_sparse()
