from time import time

import numpy as np


def benchmark(MatrixA, MatrixB, size, depth, pre_set, operation):
    matrix_a = MatrixA(size, depth)
    matrix_b = MatrixB(size, depth)
    if pre_set:
        pre_set(matrix_a)
        print(f"{MatrixA.__name__} set")
        pre_set(matrix_b)
        print(f"{MatrixB.__name__} set")

    print(f"measure {MatrixA.__name__}")
    start_time = time()
    a_ret = operation(matrix_a)  # noqa: F841
    a_duration = time() - start_time

    print(f"measure {MatrixB.__name__}")
    start_time = time()
    b_ret = operation(matrix_b)  # noqa: F841
    b_duration = time() - start_time

    # assert np.allclose(a_ret, b_ret) or a_ret is b_ret # todo this fails for some reason...
    return a_duration, b_duration


def subtest_bench_parasym_scipy(MatrixA, MatrixB):
    test_name = f"{MatrixA.__name__} vs {MatrixB.__name__}"
    print(test_name + ": running benchmark")

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
    print(test_name + ": benchmark done")


def test_bench_parasym_scipy():
    from parasymbolic_matrix import ParasymbolicMatrix
    from parasymbolic_matrix.mock_parasymbolic_matrix import MockParasymbolicMatrix
    from scipy_matrix import ScipyMatrix

    MatrixA = ParasymbolicMatrix
    MatrixB = MockParasymbolicMatrix
    MatrixC = ScipyMatrix

    subtest_bench_parasym_scipy(MatrixA, MatrixC)
    subtest_bench_parasym_scipy(MatrixA, MatrixB)


if __name__ == "__main__":
    test_bench_parasym_scipy()
