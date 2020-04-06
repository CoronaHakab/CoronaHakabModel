import tests.matrix_tests.matrix_test_and_benchmark as test_compare
from parasymbolic_matrix import ParasymbolicMatrix
from parasymbolic_matrix.mock_parasymbolic_matrix import MockParasymbolicMatrix


def test():
    test_compare.compare_and_benchmark(ParasymbolicMatrix, MockParasymbolicMatrix)


if __name__ == "__main__":
    test()
