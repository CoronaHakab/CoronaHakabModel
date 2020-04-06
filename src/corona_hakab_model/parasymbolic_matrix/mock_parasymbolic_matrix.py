from contextlib import contextmanager
from itertools import product

import numpy as np


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
