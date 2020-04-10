from functools import singledispatch


@singledispatch
def write():
    raise TypeError


try:
    from bsa.parasym import ParasymbolicMatrix, write_parasym
except ImportError:
    pass
else:
    write.register(ParasymbolicMatrix, write_parasym)

try:
    from bsa.ndarr import np, write_ndarr
except ImportError:
    pass
else:
    write.register(np.ndarray, write_ndarr)

try:
    from bsa.scipy_sparse import spmatrix, write_scipy_sparse
except ImportError:
    pass
else:
    write.register(spmatrix, write_scipy_sparse)
