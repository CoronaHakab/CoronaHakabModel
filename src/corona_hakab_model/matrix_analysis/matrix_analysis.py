from matplotlib import pyplot as plt
import numpy as np
import csv

from generation.connection_types import ConnectionTypes
from generation.matrix_generator import MatrixData
from bsa.parasym import read_parasym


class MatrixAnalyzer:
    """
    A module for analyzing and plotting statistics of the simulation matrix.
    """

    EXPORT_OUTPUT_DIR = "../../output/"
    EXPORT_HISTOGRAM_FILE = "histogram_connection_type_{}.csv"
    EXPORT_RAW_MATRIX_FILE = "raw_matrix_connection_type_{}.csv"

    def __init__(self, matrix_data_path=None):
        self.matrix_data = MatrixData()
        with open(matrix_data_path, "rb") as import_file:
            self.matrix_data.matrix = read_parasym(import_file)
        self.matrix_data.depth = len(self.matrix_data.matrix.non_zero_columns())
        self.number_of_agents = self.matrix_data.matrix.get_size()

    def export_raw_data_to_csv(self, connection_type=None):
        agents_connections = self._get_connections_from_matrix(connection_type)
        connections_values = self._get_connections_values(agents_connections, connection_type)
        if connection_type is None:
            connection_type = "all"
        headers = ['agent', 'connection', 'value']
        export_file_path = self.EXPORT_OUTPUT_DIR+self.EXPORT_RAW_MATRIX_FILE.format(connection_type)
        with open(export_file_path, 'w', newline='') as matrix_file:
            csv_writer = csv.writer(matrix_file, delimiter=',')
            csv_writer.writerows([headers]+connections_values)

    def analyze_histogram(self, connection_type=None):
        agents_connections = self._get_connections_from_matrix(connection_type)
        if connection_type is None:
            connection_type = "all"

        distribution_of_connections = [len(row) for row in agents_connections]
        average_connection = np.mean(distribution_of_connections)
        median_connection = np.median(distribution_of_connections)
        max_connection = max(distribution_of_connections)
        min_connection = min(distribution_of_connections)
        connections_histogram, _ = np.histogram(distribution_of_connections,
                                                bins=(max_connection-min_connection+1),
                                                range=(min_connection - 0.5, max_connection + 0.5))
        bins = range(min_connection, max_connection+1)
        self.export_histogram_to_csv(connections_histogram, bins,
                                     average_connection, median_connection, connection_type)
        self.plot_histogram(connections_histogram, bins, average_connection, median_connection, connection_type)

    def export_histogram_to_csv(self, histogram, bins, average_value, median_value, connection_type):
        headers = ['num_of_connections']+bins+['average', 'median', 'type']
        values = ['amount']+histogram+[average_value, median_value, connection_type]
        export_file_path = self.EXPORT_OUTPUT_DIR+self.EXPORT_HISTOGRAM_FILE.format(connection_type)
        with open(export_file_path, 'w', newline='') as histogram_file:
            csv_writer = csv.writer(histogram_file, delimiter=',')
            csv_writer.writerows([headers, values])

    def _get_connections_from_matrix(self, connection_type=None):
        raw_connection_matrix = self.matrix_data.matrix.non_zero_columns()
        if connection_type:
            agents_connections = raw_connection_matrix[connection_type]
        else:
            # if we're looking at all types of connections we want to discard multiple type of connections for
            # same agents so instead having list of lists, we will keep list of sets, so that connections are unique
            agents_connections = [set(agent_connections) for agent_connections in raw_connection_matrix[0]]
            for conn_type in range(1, self.matrix_data.depth):
                for agent_indx in range(self.number_of_agents):
                    agents_connections[agent_indx].union(raw_connection_matrix[conn_type][agent_indx])
        return agents_connections

    def _get_connections_values(self, agents_connections, connection_type):
        connections_values = []
        for agent_indx, agent_conn in enumerate(agents_connections):
            for conn in agent_conn:
                if connection_type:
                    conn_value = self.matrix_data.matrix.get(connection_type, agent_indx, conn)
                else:
                    conn_value = sum(
                        [self.matrix_data.matrix.get(conn_type, agent_indx, conn)
                         for conn_type in range(self.matrix_data.depth)
                         ]
                    )
                connections_values.append([agent_indx, conn, conn_value])
        return connections_values

    @staticmethod
    def plot_histogram(histogram, bins, average_value, median_value, connection_type):
        plt.bar(bins, histogram)
        plt.ylabel("count")
        plt.xlabel("# of connections")
        plt.xticks()
        plt.title("Histogram of connections for connection type {}\n"
                  "Avg. # of connections = {}, Med. # of connections = {}"
                  .format(connection_type, average_value, median_value))
        plt.show()
