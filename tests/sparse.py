from itertools import product
from typing import Union

import numpy as np

from consts import generator
from sparse_matrix import SparseMatrix, MagicOperator


class npMagicOp:
    def __call__(self, d, w):
        return 1 - d * w


class npSparseMatrix:
    def __init__(self, size, magic_op):
        self.size = size
        self.magic_op = magic_op
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
            if self.prob_coff_row[i] != 1:
                ret[i] *= self.prob_coff_row[i]
            if self.prob_coff_col[i] != 1:
                ret[:, i] *= self.prob_coff_col[i]
        return ret

    def manifest(self, sample=None):
        pactual = self.probs_actual()
        if sample is None:
            sample = np.random.sample(self.nz_count())
        s = iter(sample)
        inner = np.zeros_like(self.vals)
        for row, col in product(range(self.size), repeat=2):
            p = pactual[row, col]
            if p != 0 and next(s) < p:
                inner[row, col] = self.vals[row, col]

        return npManifest(self, inner)


class npManifest:
    def __init__(self, origin, inner):
        self.origin = origin
        self.inner = inner

    def I_POA(self, v):
        ret = np.ones(self.origin.size, dtype=np.float32)
        for row, col in product(range(self.origin.size), repeat=2):
            ret[row] *= self.origin.magic_op(self.inner[row, col], v[col])
        return ret

    def nz_rows(self):
        ret = []
        for row in self.inner:
            ret.append(list(np.flatnonzero(row)))
        return ret


v = np.array([0.2, 0.3, 0.6, 0], dtype=np.float32)


def check_equal(ps: SparseMatrix, mck: npSparseMatrix, msg: str):
    for i, j in product(range(mck.size), repeat=2):
        p = ps[i, j]
        m = mck[i, j]
        assert (p is m) or np.allclose(p, m), f"{msg}, [{i},{j}] {p} vs {m}"
    assert ps.nz_count() == mck.nz_count(), f"{msg}, nz"
    sample = generator.random(ps.nz_count(), dtype=np.float32)
    m = ps.manifest(sample)
    mm = mck.manifest(sample)
    i = m.I_POA(v)
    im = mm.I_POA(v)
    assert np.allclose(i,im), f"{msg}, IOP {i} vs {im}"
    mnz = m.nz_rows()
    mmnz = mm.nz_rows()
    assert mnz == mmnz,  f"{msg}, NZR {mnz} vs {mmnz}"


def operate(matrix: Union[npSparseMatrix, SparseMatrix]):
    yield 'pre_set'
    matrix.batch_set(0, [1, 2, 3], [0.2, 0.5, 1], [0.1, 0.1, 0.2])
    matrix.batch_set(1, [0, 3], [0.1, 0.1], [0.7, 1])
    matrix.batch_set(2, [0], [0.7], [0.5])
    matrix.batch_set(3, [2], [0.1], [0.9])
    yield 'post_set'
    matrix.row_set_prob_coff(2, 0.5)
    yield 'mr'
    matrix.col_set_prob_coff(0, 0)
    yield 'mc'


def test_sparse():
    magic = MagicOperator()
    main = SparseMatrix(4, magic)
    mock = npSparseMatrix(4, npMagicOp())

    m = operate(main)
    c = operate(mock)
    for i, j in zip(m, c):
        print(i)
        assert i == j
        check_equal(main, mock, i)


if __name__ == "__main__":
    test_sparse()
