from time import time
from src.timeit_print import timeit_print

import numpy as np


def benchmark_run(matrix, pre_set, operation):
    if pre_set:
        pre_set(matrix)
        print(f"{type(matrix).__name__} set")

    print(f"measure {type(matrix).__name__}")
    start_time = time()
    ret = operation(matrix)  # noqa: F841
    duration = time() - start_time

    return ret, duration


def benchmark(MatrixA, MatrixB, size, depth, pre_set, operation):
    matrix_a = MatrixA(size, depth)
    matrix_b = MatrixB(size, depth)

    a_ret, a_duration = benchmark_run(matrix_a, pre_set, operation)
    b_ret, b_duration = benchmark_run(matrix_b, pre_set, operation)

    # assert np.allclose(a_ret, b_ret) or a_ret is b_ret # todo this fails for some reason...
    return a_duration, b_duration


@timeit_print
def subtest_bench(MatrixA, MatrixB):
    t = 1000

    v = np.random.choice([0, 0.2, 0.5, 0.3, 1], t)
    v = np.asarray(v, dtype=np.float32)

    row_lens = np.arange(t) % (t // 100)
    row_indices = [np.random.permutation(t)[:rl] for rl in row_lens]
    row_values = [np.random.choice([0.1, 0.15, 0.2, 0.3], rl) for rl in row_lens]

    def build_x_3(m):
        with m.lock_rebuild():
            for c in range(3):
                for row in range(t):
                    m[c, row, row_indices[row]] = row_values[row]

    def poa(m):
        for _ in range(10_000):
            a = m.prob_any(v)
        return a

    benchmarks = ((f"poa {t}x3", (t, 3, build_x_3, poa)), (f"build {t}x3", (t, 3, None, build_x_3)))

    for name, args in benchmarks:
        p, s = benchmark(MatrixA, MatrixB, *args)
        print(f"{name}, {MatrixA.__name__}: {p}, {MatrixB.__name__}: {s}")
