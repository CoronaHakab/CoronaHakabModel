from itertools import product
import numpy as np
from src.timeit_print import timeit_print

v = np.array([0.2, 0, 0.5], dtype=np.float32)


def check_equal(matrix_a, matrix_b, msg: str):
    for i, j in product(range(3), repeat=2):
        p, m = matrix_a.get(i, j), matrix_b.get(i, j)
        assert np.isclose(p, m), f"{msg}, [{i},{j}] {p} vs {m}"
    assert np.isclose(matrix_a.total(), matrix_b.total()), msg
    p, m = matrix_a.prob_any(v), matrix_b.prob_any(v)
    assert np.allclose(p, m), msg


def operate(parasym):
    with parasym.lock_rebuild():
        parasym[0, 0, [1]] = [0.2]
        parasym[0, 1, [0]] = [0.2]
        parasym[0, 2, [1]] = [0.6]

        parasym[1, 0, [2]] = [0.1]
        parasym[1, 2, [0, 2]] = [0.5, 0.5]

    yield "post_build"

    parasym[0, 1, [1]] = [0.3]
    yield "set_no_lock"

    parasym.set_factors([0.5, 0])
    yield "set_factors"

    parasym.set_factors([1, 1])
    yield "reset_factors"

    parasym *= 12.65
    yield "mul big"

    parasym *= 0.01
    yield "mul small"

    parasym.mul_sub_row(1, 2, 0.5)
    yield "msr"

    parasym.mul_sub_col(0, 1, 0)
    yield "msc"

    parasym.reset_mul_row(1, 2)
    yield "rsr"

    parasym.reset_mul_col(0, 1)
    yield "rsc"


@timeit_print
def subtest_compare(MatrixA, MatrixB):
    main = MatrixA(3, 2)
    mock = MatrixB(3, 2)

    m = operate(main)
    c = operate(mock)
    for i, j in zip(m, c):
        print(i)
        assert i == j
        check_equal(main, mock, i)

