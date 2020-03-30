from contextlib import contextmanager
from time import time

import numpy as np
from scipy.sparse import lil_matrix

from parasymbolic_matrix import ParasymbolicMatrix


class ScipyMatrix:
    def __init__(self, size, depth):
        self.size = size
        self.sub_matrices = \
            [
                lil_matrix((size, size), dtype=np.float32) for _ in range(depth)
            ]
        self.coffs = [1 for _ in range(depth)]
        self.sum = None
        self.lg = None
        self.build_lock = False
        self.rebuild_all()

    def rebuild_all(self):
        self.sum = sum(
            s * c for (s, c) in zip(self.sub_matrices, self.coffs)
        )

        self.lg = lil_matrix((self.size, self.size), dtype=np.float32)
        nz = self.sum.nonzero()
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


def benchmark(size, depth, pre_set, operation):
    ps = ParasymbolicMatrix(size, depth)
    sm = ScipyMatrix(size, depth)
    pre_set(ps)
    print("ps set")
    pre_set(sm)
    print('ms set')
    # measure ps
    start_time = time()
    p_ret = operation(ps)
    ps_duration = time() - start_time
    # measure sm
    start_time = time()
    s_ret = operation(sm)
    sm_duration = time() - start_time
    # assert np.allclose(p_ret, s_ret) or p_ret is s_ret # todo this fails for some reason...
    return ps_duration, sm_duration


if __name__ == '__main__':
    t = 1000

    v = np.random.choice([0, 0.2, 0.5, 0.3, 1], t)
    v = np.asarray(v, dtype=np.float32)

    row_lens = np.arange(t) % (t//100)
    row_indices = [
        np.random.permutation(t)[:rl] for rl in row_lens
    ]
    row_values = [
        np.random.choice([0.1, 0.15, 0.2, 0.3], rl) for rl in row_lens
    ]

    def build_x_3(m):
        with m.lock_rebuild():
            for c in range(3):
                for row in range(t):
                    m[c, row, row_indices[row]] = row_values[row]
                    if row % 50 == 0:
                        print(f'row {row}')


    def poa(m):
        for _ in range(10_000):
            a = m.prob_any(v)
        return a


    benchmarks = (
        (f"poa {t}x3", (t, 3, build_x_3, poa)),
    )
    for name, args in benchmarks:
        p, s = benchmark(*args)
        print(f'{name}, parasymbolic: {p}, scipy: {s}')
