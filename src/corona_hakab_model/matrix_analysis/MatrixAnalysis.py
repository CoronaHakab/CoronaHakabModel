from argparse import ArgumentParser
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import csv

from generation.matrix_generator import MatrixData


class MatrixAnalyzer:
    """
    A module for analyzing and plotting statistics of the simulation matrix.
    """

    EXPORT_HISTOGRAM_DIR = "../../output/matrix_analysis/histogram_analysis/"
    EXPORT_HISTOGRAM_FILE = "histogram_connection_type_{}.csv"
    EXPORT_HISTOGRAM_IMG = "histogram_connection_type_{}.png"
    EXPORT_MATRIX_DIR = "../../output/matrix_analysis/raw_matrices/"
    EXPORT_MATRIX_FILE = "raw_matrix_connection_type_{}.csv"

    def __init__(self, matrix_data_path=None):
        self.matrix_data = MatrixData()
        self.matrix_data.import_matrix_data_as_scipy_sparse(matrix_data_path)
        self.matrix_data.depth = len(self.matrix_data.matrix)
        self._convert_to_csr()

        Path(self.EXPORT_HISTOGRAM_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.EXPORT_MATRIX_DIR).mkdir(parents=True, exist_ok=True)

    def export_raw_matrices_to_csv(self):
        """
        run over all types of connections and export each of them to a csv file
        """
        for connection_type in range(self.matrix_data.depth):
            self.export_matrix_to_csv(connection_type)
        # all connections combined
        self.export_matrix_to_csv()

    def export_matrix_to_csv(self, connection_type=None):
        if connection_type is None:
            connection_matrix = sum(self.matrix_data.matrix)
            connection_type = "all"
        else:
            connection_matrix = self.matrix_data.matrix[connection_type]

        # get the indices and values of all non-zero values in the matrix
        (connection, agent), value = connection_matrix.nonzero(), connection_matrix.data
        connections = map(lambda x, y, z: [x, y, z], agent, connection, value)
        connections = sorted(connections)

        headers = ['agent', 'connection', 'value']
        export_file_path = self.EXPORT_MATRIX_DIR + self.EXPORT_MATRIX_FILE.format(connection_type)
        with open(export_file_path, 'w', newline='') as matrix_file:
            csv_writer = csv.writer(matrix_file, delimiter=',')
            csv_writer.writerows([headers]+connections)

    def analyze_histograms(self):
        """
        run over all types of connections and analyze the connection histogram
        """
        for connection_type in range(self.matrix_data.depth):
            self.analyze_connection_histogram(connection_type)
        self.analyze_connection_histogram()

    def analyze_connection_histogram(self, connection_type=None):
        if connection_type is None:
            connection_matrix = sum(self.matrix_data.matrix)
            connection_type = "all"
        else:
            connection_matrix = self.matrix_data.matrix[connection_type]

        # get the number of connections for each agent, i.e. the number of non-zero rows for each column
        distribution_of_connections = connection_matrix.getnnz(0)

        average_connection = np.mean(distribution_of_connections)
        median_connection = np.median(distribution_of_connections)
        max_connection = np.max(distribution_of_connections)
        min_connection = np.min(distribution_of_connections)
        connections_histogram, _ = np.histogram(distribution_of_connections,
                                                bins=(max_connection-min_connection+1),
                                                range=(min_connection - 0.5, max_connection + 0.5))
        bins = np.arange(min_connection, max_connection+1)

        self.export_histogram_to_csv(connections_histogram, bins,
                                     average_connection, median_connection, connection_type)
        self.plot_histogram(connections_histogram, bins, average_connection, median_connection, connection_type)

    def export_histogram_to_csv(self, histogram, bins, average_value, median_value, connection_type):
        headers = ['num_of_connections'] + bins.tolist() + ['average', 'median', 'type']
        values = ['amount'] + histogram.tolist() + [average_value, median_value, connection_type]

        export_file_path = self.EXPORT_HISTOGRAM_DIR + self.EXPORT_HISTOGRAM_FILE.format(connection_type)
        with open(export_file_path, 'w', newline='') as histogram_file:
            csv_writer = csv.writer(histogram_file, delimiter=',')
            csv_writer.writerows([headers, values])

    def plot_histogram(self, histogram, bins, average_value, median_value, connection_type):
        plt.figure()
        plt.bar(bins, histogram, width=1)
        plt.ylabel("count")
        plt.xlabel("# of connections")
        plt.xticks()
        plt.title("Histogram of connections for connection type {}\n"
                  "Avg. # = {}  ,   Med. # = {}"
                  .format(connection_type, average_value, median_value))
        plt.savefig(self.EXPORT_HISTOGRAM_DIR + self.EXPORT_HISTOGRAM_IMG.format(connection_type))
        plt.draw()

    def _convert_to_csr(self):
        for index in range(self.matrix_data.depth):
            self.matrix_data.matrix[index] = self.matrix_data.matrix[index].tocsr()


def main():
    parser = ArgumentParser("Matrix Analyzer")
    parser.add_argument("-m",
                        "--matrix",
                        dest="matrix_path",
                        help="Matrix file to analyze")
    parser.add_argument("-s",
                        "--show",
                        dest="show",
                        action="store_true",
                        help="Show histograms")
    args = parser.parse_args()

    matrix_analyzer = MatrixAnalyzer(args.matrix_path)
    matrix_analyzer.export_raw_matrices_to_csv()
    matrix_analyzer.analyze_histograms()
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
