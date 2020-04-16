import logging
import math
import pickle
from itertools import islice
from random import random, sample
from typing import List, Iterable, Dict

import corona_matrix
import numpy as np
from corona_hakab_model_data.__data__ import __version__
from generation.circles import SocialCircle
from generation.circles_generator import PopulationData
from generation.connection_types import (
    Connect_To_All_types,
    ConnectionTypes,
    Geographic_Clustered_types,
    Random_Clustered_types,
)
from generation.matrix_consts import MatrixConsts, ConnectionTypeData
from generation.node import Node
from sparse_base import SparseBase
from encounter_layer import EncounterLayerSet, EncounterLayer
from clustered_matrix.clustered import Cluster, ClusteredSparseMatrix
from sparse_matrix.sparse import SparseMatrix
from util import rv_discrete


class MatrixData:
    __slots__ = ("matrix_type", "matrix")

    def __init__(self):
        self.matrix_type = None
        self.matrix = None

    # todo make this work, using the parasymbolic matrix serialization.
    def export(self, export_path, file_name: str):
        pass

    # todo make this work, using the parasymbolic matrix de-serialization.
    @staticmethod
    def import_matrix_data(import_file_path: str) -> "MatrixData":
        pass


# todo right now only supports parasymbolic matrix. need to merge with corona matrix class import selector
class MatrixGenerator:
    """
    this module gets the circles and agents created in circles generator and creates a matrix and sub matrices with them.
    """

    def __init__(
        self, population_data: PopulationData, matrix_consts: MatrixConsts = MatrixConsts(),
    ):
        # initiate everything
        self.logger = logging.getLogger("MatrixGenerator")
        self.matrix_data = MatrixData()
        self.matrix_consts = matrix_consts
        self._unpack_population_data(population_data)
        self.size = len(self.agents)
        self.connection_types_data: Dict[ConnectionTypes, ConnectionTypeData] = {
            con_type: matrix_consts.get_connection_type_data(con_type) for con_type in ConnectionTypes}

        self.matrix = EncounterLayerSet()
        self.matrix_type = "EncounterLayerSet"

        # create all sub matrices
        for con_type, con_type_data in self.connection_types_data.items():
            if con_type in Connect_To_All_types:
                self._create_fully_connected_layer(con_type_data, self.social_circles_by_connection_type[con_type])
            elif con_type in Random_Clustered_types:
                self._create_random_clustered_circles_matrix(
                    con_type_data, self.social_circles_by_connection_type[con_type])
            elif con_type in Geographic_Clustered_types:
                self._create_community_clustered_circles_matrix(
                    con_type_data, self.social_circles_by_connection_type[con_type])

        # current patch since matrix is un-serializable
        self.matrix_data.matrix_type = self.matrix_type
        self.matrix_data.matrix = self.matrix
        # export the matrix data
        # self.export_matrix_data()

    def _unpack_population_data(self, population_data):
        self.agents = population_data.agents
        self.social_circles_by_connection_type = population_data.social_circles_by_connection_type
        self.geographic_circles = population_data.geographic_circles
        self.geographic_circle_by_agent_index = population_data.geographic_circle_by_agent_index
        self.social_circles_by_agent_index = population_data.social_circles_by_agent_index

    def _add_sparse_matrix_layer(self, con_type_data: ConnectionTypeData, connections: List[List[int]]):
        """
        adds a new layer to the matrix, using sparse_matrix.
        :param connections: list of indices, representing a connected agent
        """
        assert len(connections) == self.size

        sparse_matrix = SparseMatrix(self.size)
        for index, connected_indices in enumerate(connections):
            connected_indices = np.sort(connected_indices)
            probs, vals = con_type_data.get_probs_and_vals(len(connected_indices))
            sparse_matrix.batch_set(index, connected_indices, probs, vals)
        magic_operator = con_type_data.magic_operator
        layer = EncounterLayer(con_type_data.connection_type, magic_operator, sparse_matrix)
        self.matrix.add_layer(layer)

    def _create_fully_connected_layer(self, con_type_data: ConnectionTypeData, circles: List[SocialCircle]):
        sparse_matrix = ClusteredSparseMatrix([[a.index for a in circle.agents] for circle in circles])
        for circle in circles:
            for agent in circle.agents:
                indeces = np.array(circle.get_indexes_of_my_circle(agent.index), dtype=int)
                if indeces.size > 1:
                    indeces = np.sort(indeces)
                probs, vals = con_type_data.get_probs_and_vals(len(indeces))
                sparse_matrix.batch_set(agent.index, indeces, probs, vals)
        magic_operator = con_type_data.magic_operator
        layer = EncounterLayer(con_type_data.connection_type, magic_operator, sparse_matrix)
        self.matrix.add_layer(layer)

    # TODO add this
    def _add_clustered_matrix_layer(self, con_type: ConnectionTypes, clusters: Iterable[Cluster]):
        """should allow a not fully connected clustered layer"""
        pass

    def _create_random_clustered_circles_matrix(self, con_type_data: ConnectionTypeData, circles: List[SocialCircle]):
        # the new connections will be saved here
        connections = [[] for _ in self.agents]
        # gets data from matrix consts
        daily_connections_float = con_type_data.daily_connections_amount
        weekly_connections_float = con_type_data.weekly_connections_amount
        total_connections_float = daily_connections_float + weekly_connections_float

        # TODO this should be switched with a better algorithm
        # adding all super small circles, into one circle, and randomly create connections inside it
        super_small_circles_combined = SocialCircle(con_type_data.connection_type)

        for circle in circles:
            agents = circle.agents
            indexes = [agent.index for agent in agents]
            nodes: List[Node] = [Node(index) for index in indexes]

            # the number of nodes. writes it for simplicity
            n = len(indexes)
            connections_amounts = iter(self.random_round(total_connections_float / 2, shape=n))
            # saves the already-inserted nodes
            np.random.shuffle(nodes)
            initial_con_amount = math.ceil(total_connections_float / 2) + 1

            # checks, if the circle is too small for any algorithm. if so adds to super small circle
            if n < initial_con_amount:
                super_small_circles_combined.add_many(circle.agents)
                continue

            # checks, if the circle is too small for normal clustering
            if n < self.matrix_consts.clustering_switching_point:
                self._add_small_circle_connections(circle, connections, total_connections_float)
                continue

            connected_nodes = set()

            # manually generate the minimum required connections
            for i in range(initial_con_amount):
                other_nodes = nodes[0:initial_con_amount]
                other_nodes.pop(i)
                nodes[i].add_connections(other_nodes)
                connected_nodes.add(nodes[i])

            # add the rest of the nodes, one at a time
            for node in islice(nodes, initial_con_amount, None):
                # selects the first node to attach to randomly
                first_connection = sample(connected_nodes, 1)[0]
                num_connections = connections_amounts.__next__()
                # fill connections other than first_connection
                while len(node.connected) < num_connections - 1:
                    if random() < con_type_data.triad_p:
                        # close the triad with a node from first_connection's connections
                        possible_nodes = first_connection.connected
                    else:
                        # connect with a node NOT from the first_connection's connections
                        possible_nodes = connected_nodes.difference(set([first_connection])).difference(
                            first_connection.connected
                        )

                    # prevent connecting a connected node
                    possible_nodes = possible_nodes.difference(node.connected)

                    # edge cases - take any node. this takes care of both sides of previous IF failing.
                    if len(possible_nodes) == 0:
                        possible_nodes = connected_nodes.difference(set([first_connection])).difference(node.connected)
                        if len(possible_nodes) == 0:
                            break

                    random_friend = sample(possible_nodes, 1)[0]
                    Node.connect(random_friend, node)
                # connect to bff here to prevent self-selection in bff's friends
                Node.connect(first_connection, node)
                connected_nodes.add(node)

            for connected_node in nodes:
                connections[connected_node.index].extend([other_node.index for other_node in connected_node.connected])

        # adding connections between all super small circles
        self._add_small_circle_connections(super_small_circles_combined, connections, total_connections_float)

        # add the current layer to the matrix
        self._add_sparse_matrix_layer(con_type_data, connections)

    def _create_community_clustered_circles_matrix(self, con_type_data: ConnectionTypeData, circles: List[SocialCircle]):
        # the new connections will be saved here
        connections = [[] for _ in self.agents]
        # gets data from matrix consts
        daily_connections_float = con_type_data.daily_connections_amount
        weekly_connections_float = con_type_data.weekly_connections_amount
        total_connections_float = daily_connections_float + weekly_connections_float

        # the number of nodes. writes it for simplicity
        connections_amounts = iter(
            MatrixGenerator.random_round(total_connections_float / 2, shape=len(self.agents))
        )

        for circle in circles:
            agents = circle.agents
            indexes = [agent.index for agent in agents]
            nodes: List[Node] = [Node(index) for index in indexes]

            if len(agents) == 0:
                continue
            # insert first node
            first_node = sample(nodes, 1)[0]
            connected_nodes = set([first_node])
            nodes.remove(first_node)
            # go over other nodes in circle
            for node in nodes:
                # first find a random node. Should never fail as we have inserted a node before.
                first_connection = sample(connected_nodes, 1)[0]

                num_connections = connections_amounts.__next__()
                # fill connections other than first_connection
                while len(node.connected) < num_connections - 1:
                    if random() < con_type_data.triad_p:
                        # close the triad with a node from first_connection's connections
                        possible_nodes = first_connection.connected
                    else:
                        # connect with a node NOT from the first_connection's connections
                        possible_nodes = connected_nodes.difference(set([first_connection])).difference(
                            first_connection.connected
                        )

                    # prevent connecting a connected node
                    possible_nodes = possible_nodes.difference(node.connected)

                    # edge cases - take any node. this takes care of both sides of previous IF failing.
                    if len(possible_nodes) == 0:
                        possible_nodes = connected_nodes.difference(set([first_connection])).difference(node.connected)
                        if len(possible_nodes) == 0:
                            break

                    random_friend = sample(possible_nodes, 1)[0]
                    Node.connect(random_friend, node)
                # connect to bff here to prevent self-selection in bff's friends
                Node.connect(first_connection, node)
                connected_nodes.add(node)

            nodes.append(first_node)
            for connected_node in nodes:
                connections[connected_node.index].extend([other_node.index for other_node in connected_node.connected])

        # add the current layer to the matrix
        self._add_sparse_matrix_layer(con_type_data, connections)

    # todo when the amount of people in the circle is vary small, needs different solution
    def _add_small_circle_connections(self, circle: SocialCircle, connections: List[List], scale_factor: float):
        """
        used to create the connections for circles too small for the clustering algorithm.
        creates circle's connections, and adds them to a given connections list
        :param circle: the social circle too small
        :param connections: the connections list
        :param scale_factor: average amount of connections for each agent
        :return:
        """
        remaining_contacts = {
            agent.index: math.ceil(np.random.exponential(scale_factor - 0.5)) for agent in circle.agents
        }

        agent_id_pool = set([agent.index for agent in circle.agents])

        # while there are still connections left to make
        while len(agent_id_pool) >= 2:
            current_agent_id = agent_id_pool.pop()

            rc = min(remaining_contacts[current_agent_id], len(agent_id_pool))
            conns = np.array(sample(agent_id_pool, rc))
            connections[current_agent_id].extend(conns)
            for other_agent_id in conns:
                connections[other_agent_id].append(current_agent_id)
            to_remove = set()
            for id in conns:
                remaining_contacts[id] -= 1
                if remaining_contacts[id] == 0:
                    to_remove.add(id)
            assert to_remove <= agent_id_pool

            agent_id_pool.difference_update(to_remove)

    @staticmethod
    def random_round(x: float, shape: int = 1):
        """
        randomly chooses between floor and ceil such that the average will be x
        :param x: a float
        :param shape: amount of wanted rolls
        :return: numpy array of ints, each is either floor or ceil
        """
        if x % 1 == 0:
            return np.full(shape, x, dtype=int)
        floor_prob = math.ceil(x) - x
        ceil_prob = x - math.floor(x)
        return np.random.choice([math.floor(x), math.ceil(x)], size=shape, p=[floor_prob, ceil_prob])
