from contextlib import contextmanager
from io import BytesIO
from itertools import product

import numpy as np

from bsa.parasym import write_parasym, read_parasym
from parasymbolic_matrix import ParasymbolicMatrix


class MockParasymbolicMatrix:
    class MockCoffMatrix:
        def __init__(self, size):
            self.arr = np.zeros((size, size), dtype=np.float32)
            self.row_coffs = np.ones(size, dtype=np.float32)
            self.col_coffs = np.ones(size, dtype=np.float32)
            self.size = size

        def get(self, i, j):
            return self.arr[i, j] * self.row_coffs[i] * self.col_coffs[j]

        def total(self):
            return sum(self.get(i, j) for i, j in product(range(self.size), repeat=2))

    def __init__(self, size, components):
        self.components = [self.MockCoffMatrix(size) for _ in range(components)]
        self.coffs = [1 for _ in range(components)]

        self.size = size

    def get(self, arg1, arg2, arg3=None):
        if arg3 is None:
            return sum(c.get(arg1, arg2) * f for (c, f) in zip(self.components, self.coffs))
        return self.components[arg1].get(arg2, arg3)

    def total(self):
        return sum(c.total() * f for (c, f) in zip(self.components, self.coffs))

    def prob_any(self, v):
        ret = []
        v = np.asanyarray(v)
        for row in range(self.size):
            i_r = 1
            for col in range(self.size):
                i_r *= 1 - v[col] * self.get(row, col)
            ret.append(1 - i_r)
        return ret

    def __imul__(self, other):
        self.coffs = [c * other for c in self.coffs]
        return self

    def set_factors(self, f):
        self.coffs = f

    def mul_sub_row(self, comp, row, factor):
        self.components[comp].row_coffs[row] *= factor

    def mul_sub_col(self, comp, col, factor):
        self.components[comp].col_coffs[col] *= factor

    def reset_mul_row(self, comp, row):
        self.components[comp].row_coffs[row] = 1

    def reset_mul_col(self, comp, col):
        self.components[comp].col_coffs[col] = 1

    def __setitem__(self, key, value):
        comp, row, indices = key
        self.components[comp].arr[row][indices] = value

    @contextmanager
    def lock_rebuild(self):
        yield self


v = np.array([0.2, 0, 0.5], dtype=np.float32)


def check_equal(ps: ParasymbolicMatrix, mck: MockParasymbolicMatrix, msg: str):
    for i, j in product(range(3), repeat=2):
        p, m = ps.get(i, j), mck.get(i, j)
        assert np.isclose(p, m), f"{msg}, [{i},{j}] {p} vs {m}"
    assert np.isclose(ps.total(), mck.total()), msg
    p, m = ps.prob_any(v), mck.prob_any(v)
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


def test_parasym():
    main = ParasymbolicMatrix(3, 2)
    mock = MockParasymbolicMatrix(3, 2)

    m = operate(main)
    c = operate(mock)
    for i, j in zip(m, c):
        print(i)
        assert i == j
        check_equal(main, mock, i)


def test_read_write():
    arr = ParasymbolicMatrix(3, 2)
    with arr.lock_rebuild():
        arr[0, 0, [1]] = [0.2]
        arr[0, 1, [0]] = [0.2]
        arr[0, 2, [1]] = [0.6]

        arr[1, 0, [2]] = [0.1]
        arr[1, 2, [0, 2]] = [0.5, 0.5]

    sink = BytesIO()
    write_parasym(arr, sink)
    sink.seek(0)
    dec = read_parasym(sink)

    for i, j in product(range(3), repeat=2):
        for layer in range(2):
            assert arr.get(layer, i, j) == dec.get(layer, i, j)
        assert arr.get(i, j) == dec.get(i, j)


if __name__ == "__main__":
    test_parasym()
