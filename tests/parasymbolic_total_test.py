import numpy as np

from generation.circles_consts import CirclesConsts
from generation.generation_manager import GenerationManger
from generation.matrix_consts import MatrixConsts
from generation.connection_types import ConnectionTypes
from parasymbolic_matrix import ParasymbolicMatrix


def create_fully_connected_population_matrix(population_size=3000) -> ParasymbolicMatrix:
    cc = CirclesConsts(
        population_size=population_size,
        connection_type_prob_by_age_index=[
            {
                ConnectionTypes.Work: 0,
                ConnectionTypes.School: 0.0,
                ConnectionTypes.Family: 1.0,
                ConnectionTypes.Other: 0.0,
            },
            {
                ConnectionTypes.Work: 0,
                ConnectionTypes.School: 0,
                ConnectionTypes.Family: 1.0,
                ConnectionTypes.Other: 0.0,
            },
            {
                ConnectionTypes.Work: 0,
                ConnectionTypes.School: 0,
                ConnectionTypes.Family: 1.0,
                ConnectionTypes.Other: 0.0,
            },
        ],
        circle_size_distribution_by_connection_type={
            ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
            ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
            ConnectionTypes.Family: ([population_size + 1], [1]),
            ConnectionTypes.Other: ([population_size * 2], [1.0]),
        },
        geo_circles_amount=1,
        geo_circles_names=["north"],
        geo_circles_agents_share=[1],
        multi_zone_connection_type_to_geo_circle_probability=[
            {ConnectionTypes.Work: {"North": 1}}
        ]
    )
    mc = MatrixConsts(use_parasymbolic_matrix=True)
    gm = GenerationManger(circles_consts=cc, matrix_consts=mc)
    return gm.matrix_data.matrix


def compare_total_using_np(matrix: ParasymbolicMatrix):
    np_sum = np.sum(np.fromiter((matrix.get(i, j) for i in range(matrix.get_size()) for j in range(matrix.get_size())), dtype=float))
    matrix_sum = matrix.total()
    print(f"matrix total: {matrix_sum}, np sum: {np_sum}")
    print(f" difference = {matrix_sum - np_sum}")


def compare_total_using_loop(matrix: ParasymbolicMatrix):
    loop_sum = 0
    non_zero_columns = matrix.non_zero_columns()
    for depth, rows in enumerate(non_zero_columns):
        for row, columns in enumerate(rows):
            for column in columns:
                loop_sum += matrix.get(depth, row, column)
    matrix_sum = matrix.total()
    print(f"matrix total: {matrix_sum}, loop_sum: {loop_sum}")
    print(f" difference = {matrix_sum - loop_sum}")


def analyze_fully_connected_matrix():
    matrix = create_fully_connected_population_matrix()
    compare_total_using_np(matrix)

if __name__ == "__main__":
    analyze_fully_connected_matrix()
