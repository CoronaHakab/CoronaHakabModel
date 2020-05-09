from argparse import ArgumentParser
from itertools import zip_longest
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import csv
from scipy.sparse import lil_matrix

from generation.matrix_generator import MatrixData
from generation.connection_types import ConnectionTypes
import project_structure

# export files
EXPORT_HISTOGRAM_FILE = "histogram_connection_type_{}.csv"
EXPORT_HISTOGRAM_IMG = "histogram_connection_type_{}.png"
EXPORT_MATRIX_FILE = "raw_matrix_connection_type_{}.csv"

# create output directories
EXPORT_HISTOGRAM_DIR = Path(project_structure.OUTPUT_FOLDER / "matrix_analysis" / "histogram_analysis")
EXPORT_MATRIX_DIR = Path(project_structure.OUTPUT_FOLDER / "matrix_analysis" / "raw_matrices")


def import_matrix_as_csr(matrix_data_path):
    matrix_data = MatrixData.import_matrix_data(matrix_data_path, keep_matrix_lazy_evaluated=True)

    # Fill the data into lil_matrix because it's faster for assignments than csr
    lil_matrices = [lil_matrix((matrix_data.size, matrix_data.size)) for _ in range(matrix_data.depth)]
    for depth, index, conns, vals in matrix_data.matrix_assignment_data:
        lil_matrices[depth][index, conns] = vals

    # Convert lil_matrix to csr_matrix because it's easier to work with
    csr_matrices = [lil.tocsr() for lil in lil_matrices]

    return csr_matrices


def export_raw_matrices_to_csv(matrices):
    """
    Run over all types of connections and export each of them to a csv file
    """
    for layer_ind, conn_matrix in enumerate(matrices):
        conn_name = ConnectionTypes(layer_ind).name
        export_matrix_to_csv(conn_matrix, conn_name)
    # all connections combined
    conn_matrix = sum(matrices)
    export_matrix_to_csv(conn_matrix, 'All')


def export_matrix_to_csv(connection_matrix, conn_name):
    # get the indices and values of all non-zero values in the matrix
    (connection, agent), value = connection_matrix.nonzero(), connection_matrix.data
    connections = map(lambda x, y, z: [x, y, z], agent, connection, value)
    connections = sorted(connections)

    headers = ['agent', 'connection', 'value']
    EXPORT_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    export_file_path = EXPORT_MATRIX_DIR / EXPORT_MATRIX_FILE.format(conn_name)
    with open(export_file_path, 'w', newline='') as matrix_file:
        csv_writer = csv.writer(matrix_file, delimiter=',')
        csv_writer.writerows([headers] + connections)


def analyze_histograms(matrices):
    """
    Run over all types of connections and analyze the connection histogram
    """
    histograms = []
    # analyze each connection type
    for layer_ind, connection_matrix in enumerate(matrices):
        conn_name = ConnectionTypes(layer_ind).name
        histograms.append(analyze_connection_histogram(connection_matrix, conn_name))
    # analyze all connection combined
    connection_matrix = sum(matrices)
    histograms.append(analyze_connection_histogram(connection_matrix, 'All'))
    return histograms


def analyze_connection_histogram(connection_matrix, conn_name):
    # get the number of connections for each agent, i.e. the number of non-zero rows for each column
    distribution_of_connections = connection_matrix.getnnz(0)

    average_conn = np.mean(distribution_of_connections)
    median_conn = np.median(distribution_of_connections)
    max_connection = max(distribution_of_connections)
    min_connection = min(distribution_of_connections)
    connections_histogram, _ = np.histogram(distribution_of_connections,
                                            bins=(max_connection-min_connection+1),
                                            range=(min_connection - 0.5, max_connection + 0.5))
    bins = np.arange(min_connection, max_connection+1)
    return {'name': conn_name,  'number_of_connections': bins, 'count': connections_histogram,
            'average': average_conn, 'median': median_conn}


def export_histograms(histograms):
    for hist in histograms:
        export_histogram_to_csv(hist['name'], hist['count'], hist['number_of_connections'],
                                hist['average'], hist['median'])


def export_histogram_to_csv(name, histogram, bins, average_value, median_value):
    headers = ['number  of connections', 'count', 'average', 'median']
    values = zip_longest(bins.tolist(), histogram.tolist(), [average_value], [median_value])
    EXPORT_HISTOGRAM_DIR.mkdir(parents=True, exist_ok=True)
    export_file_path = EXPORT_HISTOGRAM_DIR / EXPORT_HISTOGRAM_FILE.format(name)
    with open(export_file_path, 'w', newline='') as histogram_file:
        csv_writer = csv.writer(histogram_file, delimiter=',')
        csv_writer.writerow(headers)
        csv_writer.writerows(values)


def save_histogram_plots(histograms):
    for hist in histograms:
        plt.figure()
        plt.bar(hist['number_of_connections'], hist['count'], width=1)
        plt.ylabel("count")
        plt.xlabel("# of connections")
        plt.xticks()
        plt.title("Histogram of connections for connection type {}\n"
                  "Avg. # = {}  ,   Med. # = {}"
                  .format(hist['name'], hist['average'], hist['median']))
        plt.savefig(EXPORT_HISTOGRAM_DIR / EXPORT_HISTOGRAM_IMG.format(hist['name']))
        plt.draw()


def main():
    parser = ArgumentParser("Matrix Analyzer")
    parser.add_argument("--matrix",
                        dest="matrix_path",
                        help="Matrix file to analyze")
    parser.add_argument("--show",
                        dest="show",
                        action="store_true",
                        help="Show histograms")
    args = parser.parse_args()

    matrices = import_matrix_as_csr(args.matrix_path)
    export_raw_matrices_to_csv(matrices)
    histograms = analyze_histograms(matrices)
    export_histograms(histograms)
    save_histogram_plots(histograms)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
