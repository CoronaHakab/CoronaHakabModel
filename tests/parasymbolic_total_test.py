import numpy as np

from parasymbolic_matrix import ParasymbolicMatrix


def create_fully_connected_population_matrix(population_size=300) -> ParasymbolicMatrix:
    matrix = ParasymbolicMatrix(population_size, 1)
    cols = np.arange(population_size, dtype=np.uint64)
    v = np.full(population_size, 0.001, dtype=np.float32)
    with matrix.lock_rebuild():
        for i in range(population_size):
            matrix.batch_set(0, i, cols, v)
    return matrix


def compare_total_using_np(matrix: ParasymbolicMatrix):
    np_sum = np.sum(
        np.fromiter((matrix.get(i, j) for i in range(matrix.get_size()) for j in range(matrix.get_size())),
                    dtype=float))
    matrix_sum = matrix.total()
    assert np.isclose(np_sum, matrix_sum)


def test_sum_connected():
    matrix = create_fully_connected_population_matrix()
    compare_total_using_np(matrix)


if __name__ == "__main__":
    test_sum_connected()
