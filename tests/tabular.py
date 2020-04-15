from enum import Enum, auto

import numpy as np

from sparse_matrix import WMaskedTabularOp, SparseMatrix


def test_tabular():
    class A(Enum):
        x = auto()
        y = auto()
        z = auto()

    class B(Enum):
        s = auto()
        t = auto()

    def sol(v, a, b):
        if a == A.y:
            v *= 2
        elif a == A.z:
            v *= 3

        if b == B.t:
            v **= 2
        return v

    w_dict, magic = WMaskedTabularOp.from_enums(100, 100, A, B, solutions=sol)
    w = w_dict[(A.y, B.t)]
    assert magic.operate(w, 12.6) == ((12 * 2) ** 2)
    magic = magic.mul(2)
    assert magic.operate(w, 12.6) == ((12 * 2) ** 2) * 2
    matrix = SparseMatrix(3)

    def my_batch_set(row, cols, As, Bs):
        probs = [1.0 for _ in cols]
        vals = [w_dict[a, b] for a, b in zip(As, Bs)]
        matrix.batch_set(row, cols, probs, vals)

    my_batch_set(0, [1], [A.x], [B.s])
    my_batch_set(1, [0, 1], [A.y, A.y], [B.s, B.t])
    my_batch_set(2, [1, 2], [A.z, A.x], [B.s, B.t])

    v = np.array([5, 10, 3], dtype=np.float32)
    u = matrix.manifest().I_POA(v, magic)
    # yes In did the math by hand
    assert np.allclose(u,
                       [20, 16000, 1080])


if __name__ == '__main__':
    test_tabular()
