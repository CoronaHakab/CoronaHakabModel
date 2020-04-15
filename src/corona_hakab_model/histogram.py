from collections import Counter
import numpy as np
from typing import List

from generation.connection_types import ConnectionTypes


class Histogram:
    """
    This class implements a cumulative normalized histogram, such that each time new data is received,
    a new normalized histogram is calculated, based on all the accumulated data.
    The histogram is represented as a dictionary with the following properties:
     - histogram dict:
         * 'number_of_connections': The x-axis of the histogram, number of connections for each agent
         * 'probability_density': The y-axis of the histogram, the probability of an agent the have that number of
                                  connections.
         * 'average': The average number of connections for all agents.
         * 'median': The median number of connections for all agents.
    """
    def __init__(self, connections_distribution: List = None):
        self.weighted_connections = Counter()
        self.histogram = {}
        self.min_connection = 0
        self.max_connection = 0
        if connections_distribution is not None:
            self.update(connections_distribution)

    def get(self):
        return self.histogram

    def update(self, connections_distribution: List):
        """
        Weighted_connections is a Counter object where the keys are the number of connections for an agent,
        and the values are the weight of that number, i.e. the total number of agents with that number if connections
        """
        self.weighted_connections += Counter(connections_distribution)
        self.min_connection = min(self.weighted_connections.keys())
        self.max_connection = max(self.weighted_connections.keys())
        probability_density, bins_edges = np.histogram(list(self.weighted_connections.keys()),
                                                       bins=(self.max_connection - self.min_connection + 1),
                                                       range=(self.min_connection - 0.5, self.max_connection + 0.5),
                                                       weights=list(self.weighted_connections.values()),
                                                       density=True)
        bins_centers = (bins_edges+0.5)[:-1].tolist()
        self.histogram['number_of_connections'] = bins_centers
        self.histogram['probability_density'] = probability_density.tolist()
        self.histogram['average'] = np.average(list(self.weighted_connections.keys()),
                                               weights=list(self.weighted_connections.values()))
        # The median is where the cumulative probability is 50%
        self.histogram['median'] = bins_centers[np.argmax(probability_density.cumsum() >= 0.5)]


class TimeHistograms:
    """
    This class implements time-dependent histograms, by separating the connections is the matrix to daily and weekly
    connections. The histograms are calculated for each connection-type separately and for all types together.
    The histograms are being saved to the time_histograms dict in the following way:
    - key=<histogram name>, e.g. 'daily_hist_conn_type_Work'
    - value=<Histogram object>
    """
    TIME_HISTOGRAM_NAME = '{}_hist_conn_type_{}'

    def __init__(self, matrix=None):
        self.time_histograms = {}
        self.daily_connections = []
        self.weekly_connections = []
        self.depth = len(ConnectionTypes)
        if matrix is not None:
            self.update_all_histograms(matrix)

    def update_all_histograms(self, matrix):
        self.update_time_connections(matrix)
        self.update_time_histograms(self.daily_connections, 'daily')
        self.update_time_histograms(self.weekly_connections, 'weekly')

    def update_time_connections(self, matrix):
        """
        This method gets a full connection matrix and extracts daily and weekly connection arrays,
        i.e. only if a daily/weekly connection exists, without it's value.
        I assume that each connection is either daily or weekly and that daily connections are larger than weekly ones
        for each connection type.
        I also assume that each connection type has a constant value, and that if all the connections are the same, the
        matrix is daily (for Family connection).
        """
        self.daily_connections = []
        self.weekly_connections = []
        for conn_type in range(self.depth):
            conn_mat = matrix[conn_type].tocsr()
            th = 0.5 * (np.max(conn_mat.data) + np.min(conn_mat.data))
            daily_conn_array = (conn_mat >= th).getnnz(0)
            all_conn_array = conn_mat.getnnz(0)
            self.daily_connections.append(daily_conn_array)
            self.weekly_connections.append(all_conn_array-daily_conn_array)

    def update_time_histograms(self, time_connections, time):
        """
        Update all histograms for the given time (daily/weekly)
        """
        for conn_type in range(self.depth):
            conn_name = ConnectionTypes(conn_type).name
            connections = time_connections[conn_type]
            self.update_conn_type_histogram(connections, time, conn_name)
        connections = sum(time_connections)
        self.update_conn_type_histogram(connections, time, 'all')

    def update_conn_type_histogram(self, connections, time: str, conn_name: str):
        """
        Update connection-type histogram for the given time
        """
        hist_name = self.TIME_HISTOGRAM_NAME.format(time, conn_name)
        conn_list = connections.tolist()
        if hist_name not in self.time_histograms:
            # create new histogram
            self.time_histograms[hist_name] = Histogram(conn_list)
        else:
            self.time_histograms[hist_name].update(conn_list)

    def get(self):
        return [{hist_name: histogram.get()} for hist_name, histogram in self.time_histograms.items()]
