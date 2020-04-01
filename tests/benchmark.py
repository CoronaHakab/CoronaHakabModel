from contextlib import contextmanager
from time import time

import numpy as np

from parasymbolic_matrix import ParasymbolicMatrix
from scipy_matrix import ScipyMatrix


def benchmark(size, depth, pre_set, operation):
    ps = ParasymbolicMatrix(size, depth)
    sm = ScipyMatrix(size, depth)
    if pre_set:
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


    def poa(m):
        for _ in range(10_000):
            a = m.prob_any(v)
        return a


    benchmarks = (
        (f"poa {t}x3", (t, 3, build_x_3, poa)),
        (f"build {t}x3", (t, 3, None, build_x_3))
    )
    for name, args in benchmarks:
        p, s = benchmark(*args)
        print(f'{name}, parasymbolic: {p}, scipy: {s}')
