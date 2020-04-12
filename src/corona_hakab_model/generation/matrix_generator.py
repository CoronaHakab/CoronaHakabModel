import logging
import math
from itertools import islice
from random import random, sample
from typing import List

import corona_matrix
import numpy as np
from generation.circles import SocialCircle
from generation.circles_generator import PopulationData
from generation.connection_types import (
    Connect_To_All_types,
    ConnectionTypes,
    Geographic_Clustered_types,
    Random_Clustered_types,
)
from generation.matrix_consts import MatrixConsts
from generation.node import Node
from bsa.parasym import write_parasym, read_parasym
from bsa.scipy_sparse import read_scipy_sparse


class MatrixData:
    __slots__ = ("matrix_type", "matrix", "depth")

    # import/export variables
    IMPORT_MATRIX_PATH = "../../output/matrix_data"
    EXPORT_OUTPUT_DIR = "../../output/"
    EXPORT_FILE_NAME = "matrix_data"

    def __init__(self):
        self.matrix_type = None
        self.matrix = None
        self.depth = 0

    def import_matrix_data_as_scipy_sparse(self, matrix_data_path):
        """
        Import a MatrixData object from file.
        The matrix is imported as scipy_sparse.
        """
        if matrix_data_path is None:
            matrix_data_path = self.IMPORT_MATRIX_PATH
        try:
            with open(matrix_data_path, "rb") as import_file:
                self.matrix = read_scipy_sparse(import_file)
            self.matrix_type = "scipy_sparse"
            self.depth = len(self.matrix)
        except FileNotFoundError:
            raise FileNotFoundError("Couldn't open matrix data")

    def import_matrix_data_as_parasym(self, matrix_data_path=None):
        """
        Import a MatrixData object from file.
        The matrix is imported as parasymbolic.
        """
        if matrix_data_path is None:
            matrix_data_path = self.IMPORT_MATRIX_PATH
        try:
            with open(matrix_data_path, "rb") as import_file:
                self.matrix = read_parasym(import_file)
            self.matrix_type = "parasymbolic"
            self.depth = len(self.matrix)
        except FileNotFoundError:
            raise FileNotFoundError("Couldn't open matrix data")

    def export(self, output_path=None):
        """
        This method exports the MatrixData object to a file.
        The matrix is exported as parasymbolic using the write_parasym() function.
        """
        if output_path is None:
            output_path = self.EXPORT_OUTPUT_DIR + self.EXPORT_FILE_NAME
        with open(output_path, 'wb') as export_file:
            write_parasym(self.matrix, export_file)


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
        self.depth = len(ConnectionTypes)

        CoronaMatrix = corona_matrix.get_corona_matrix_class(matrix_consts.use_parasymbolic_matrix)
        self.logger.info("Using CoronaMatrix of type {}".format(CoronaMatrix.__name__))
        self.matrix = CoronaMatrix(self.size, self.depth)

        # create all sub matrices
        with self.matrix.lock_rebuild():
            # todo switch the depth logic, to get a connection type instead of int depth
            current_depth = 0

            for con_type in ConnectionTypes:
                if con_type in Connect_To_All_types:
                    self._create_fully_connected_circles_matrix(
                        con_type, self.social_circles_by_connection_type[con_type], current_depth
                    )
                elif con_type in Random_Clustered_types:
                    self._create_random_clustered_circles_matrix(
                        con_type, self.social_circles_by_connection_type[con_type], current_depth
                    )
                elif con_type in Geographic_Clustered_types:
                    self._create_community_clustered_circles_matrix(
                        con_type, self.social_circles_by_connection_type[con_type], current_depth
                    )
                current_depth += 1

        # current patch since matrix is un-serializable
        self.matrix_data.matrix_type = "parasymbolic"
        self.matrix_data.matrix = self.matrix
        self.matrix_data.depth = self.depth
        # export the matrix data
        self.matrix_data.export()
        self.logger.info("Exported matrix")

    def _unpack_population_data(self, population_data):
        self.agents = population_data.agents
        self.social_circles_by_connection_type = population_data.social_circles_by_connection_type
        self.geographic_circles = population_data.geographic_circles
        self.geographic_circle_by_agent_index = population_data.geographic_circle_by_agent_index
        self.social_circles_by_agent_index = population_data.social_circles_by_agent_index

    def _create_fully_connected_circles_matrix(self, con_type: ConnectionTypes, circles: List[SocialCircle], depth):
        connection_strength = self.matrix_consts.connection_type_to_connection_strength[con_type]
        for circle in circles:
            ids = np.array([a.index for a in circle.agents])
            vals = np.full_like(ids, connection_strength, dtype=np.float32)
            for i, row in enumerate(ids):
                temp = vals[i]
                vals[i] = 0
                self.matrix[depth, int(row), ids] = vals
                vals[i] = temp

    def _create_random_clustered_circles_matrix(self, con_type: ConnectionTypes, circles: List[SocialCircle], depth):
        # the new connections will be saved here
        connections = [[] for _ in self.agents]
        # gets data from matrix consts
        connection_strength = self.matrix_consts.connection_type_to_connection_strength[con_type]
        daily_connections_float = self.matrix_consts.daily_connections_amount_by_connection_type[con_type]
        weekly_connections_float = self.matrix_consts.weekly_connections_amount_by_connection_type[con_type]
        total_connections_float = daily_connections_float + weekly_connections_float

        # adding all super small circles, into one circle, and randomly create connections inside it
        super_small_circles_combined = SocialCircle(con_type)

        for circle in circles:
            agents = circle.agents
            indexes = [agent.index for agent in agents]
            nodes: List[Node] = [Node(index) for index in indexes]

            # the number of nodes. writes it for simplicity
            n = len(indexes)
            connections_amounts = iter(
                self.random_round(((daily_connections_float + weekly_connections_float) / 2), shape=n)
            )
            # saves the already-inserted nodes
            inserted_nodes = set()
            np.random.shuffle(nodes)
            con_amount = math.ceil((daily_connections_float + weekly_connections_float) / 2) + 1

            # checks, if the circle is too small for any algorithm. if so adds to super small circle
            if con_amount > n:
                super_small_circles_combined.add_many(circle.agents)
                continue

            # checks, if the circle is too small for normal clustering
            if n < self.matrix_consts.clustering_switching_point[0]:
                self._add_small_circle_connections(circle, connections, total_connections_float)
                continue

            # manually generate the minimum required connections
            for i in range(con_amount):
                other_nodes = nodes[0:con_amount]
                other_nodes.pop(i)
                nodes[i].add_connections(other_nodes)
                inserted_nodes.add(nodes[i])

                # add the newly made connections to the connections list. note that this is one directional,
                # but the other direction will be added when adding the other's connections
                connections[nodes[i].index].extend([node.index for node in other_nodes])

            # add the rest of the nodes, one at a time
            for node in islice(nodes, con_amount, None):
                connections_amount = connections_amounts.__next__()
                # selects the first node to attach to randomly
                rand_node = sample(inserted_nodes, 1)[0]
                inserted_nodes.remove(rand_node)

                # adds a connection between the nodes
                connections[node.index].append(rand_node.index)
                connections[rand_node.index].append(node.index)

                # todo change this to use p, and not only p = 1
                # randomly choose the rest of the connections from rand_node connections.
                nodes_to_return = []
                for _ in range(connections_amount - 1):
                    # randomly choose a node from rand_node connections
                    new_rand = rand_node.pop_random()
                    nodes_to_return.append(new_rand)
                    Node.connect(node, new_rand)

                    # add connection to connections list
                    connections[node.index].append(new_rand.index)
                    connections[new_rand.index].append(node.index)
                # connect current node with rand node. note that this only happens here to not pick yourself later on
                Node.connect(node, rand_node)
                # return the popped nodes back to rand_node connections, and rand node back to already inserted list
                rand_node.add_connections(nodes_to_return)
                inserted_nodes.add(rand_node)
                inserted_nodes.add(node)

        # adding connections between all super small circles
        self._add_small_circle_connections(super_small_circles_combined, connections, total_connections_float)

        # insert all connections to matrix
        for agent, conns in zip(self.agents, connections):
            conns = np.array(conns)
            conns.sort()
            # rolls for each connection, whether it is daily or weekly
            daily_share = daily_connections_float / total_connections_float
            weekly_share = weekly_connections_float / total_connections_float
            strengthes = np.random.choice(
                [connection_strength, connection_strength / 7], size=len(conns), p=[daily_share, weekly_share]
            )
            v = np.full_like(conns, strengthes, dtype=np.float32)
            self.matrix[depth, agent.index, conns] = v

    def _create_community_clustered_circles_matrix(self, con_type: ConnectionTypes, circles: List[SocialCircle], depth):
        # the new connections will be saved here
        connections = [[] for _ in self.agents]
        # gets data from matrix consts
        connection_strength = self.matrix_consts.connection_type_to_connection_strength[con_type]
        daily_connections_float = self.matrix_consts.daily_connections_amount_by_connection_type[con_type]
        weekly_connections_float = self.matrix_consts.weekly_connections_amount_by_connection_type[con_type]
        total_connections_float = daily_connections_float + weekly_connections_float

        # the number of nodes. writes it for simplicity
        connections_amounts = iter(
            MatrixGenerator.random_round(
                ((daily_connections_float + weekly_connections_float) / 2), shape=len(self.agents)
            )
        )

        for circle in circles:
            agents = circle.agents
            indexes = [agent.index for agent in agents]
            nodes: List[Node] = [Node(index) for index in indexes]

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
                    if random() < self.matrix_consts.community_triad_probability[0]:
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

        # insert all connections to matrix
        for agent, conns in zip(self.agents, connections):
            conns = np.array(conns)
            conns.sort()
            # rolls for each connection, whether it is daily or weekly
            strengthes = np.random.choice(
                [connection_strength, connection_strength / 7],
                size=len(conns),
                p=[
                    daily_connections_float / total_connections_float,
                    weekly_connections_float / total_connections_float,
                ],
            )
            v = np.full_like(conns, strengthes, dtype=np.float32)
            self.matrix[depth, agent.index, conns] = v

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
        floor_prob = math.ceil(x) - x
        ceil_prob = x - math.floor(x)
        return np.random.choice([math.floor(x), math.ceil(x)], size=shape, p=[floor_prob, ceil_prob])
