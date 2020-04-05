def get_corona_matrix_class(preffer_parasymbolic):
    if preffer_parasymbolic:
        try:
            from parasymbolic_matrix import ParasymbolicMatrix as CoronaMatrix
        except ImportError:
            pass
        else:
            return CoronaMatrix
    from scipy_matrix import ScipyMatrix as CoronaMatrix

    return CoronaMatrix
