import tests.matrix_tests.matrix_test_and_benchmark as test_compare
from parasymbolic_matrix import ParasymbolicMatrix
from scipy_matrix import ScipyMatrix


def test():
    test_compare.compare_and_benchmark(ParasymbolicMatrix, ScipyMatrix)


if __name__ == "__main__":
    test()
