from contextlib import contextmanager

import numpy as np
from scipy.sparse import lil_matrix


class ScipyMatrix:
    def __init__(self, size, depth):
        self.size = size
        self.sub_matrices = [lil_matrix((size, size), dtype=np.float32) for _ in range(depth)]
        self.coffs = [1 for _ in range(depth)]
        self.sum = None
        self.lg = None
        self.build_lock = False
        self.rebuild_all()

    def rebuild_all(self):
        self.sum = sum(s * c for (s, c) in zip(self.sub_matrices, self.coffs))

        self.lg = lil_matrix((self.size, self.size), dtype=np.float32)
        # scipy.sparse.csr_matrix.nonzero
        # Returns a tuple of arrays (row,col) containing the indices of the non-zero elements of the matrix.
        nz = self.sum.nonzero()
        # updates the log matrix if non zeros are found
        if len(nz[0]):
            self.lg[nz] = np.log(1 - self.sum[nz])

    def total(self):
        return self.sum.sum()

    def prob_any(self, v):
        ret = self.lg.dot(v)
        return 1 - np.exp(ret)

    def __imul__(self, other):
        self.coffs = [c * other for c in self.coffs]
        if not self.build_lock:
            nz = self.sum.nonzero()
            self.lg = lil_matrix((self.size, self.size), dtype=np.float32)
            if len(nz[0]):
                self.sum[nz] = other * self.sum[nz]
                self.lg[nz] = np.log(1 - self.sum[nz])
            return self

    def set_factors(self, f):
        self.coffs = f
        if not self.build_lock:
            self.rebuild_all()

    def mul_sub_row(self, comp, row, factor):
        self.sub_matrices[comp][row] *= factor
        if not self.build_lock:
            self.rebuild_all()

    def mul_sub_col(self, comp, col, factor):
        self.sub_matrices[comp][:, col] *= factor
        if not self.build_lock:
            self.rebuild_all()

    def __setitem__(self, key, value):
        comp, row, indices = key
        if not len(indices):
            return
        row = np.full(len(indices), row)
        self.sub_matrices[comp][row, indices] = value
        if not self.build_lock:
            self.rebuild_all()

    @contextmanager
    def lock_rebuild(self):
        self.build_lock = False
        yield self
        self.build_lock = True
        self.rebuild_all()
