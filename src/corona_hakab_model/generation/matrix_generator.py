from __future__ import annotations

import logging
import math
import os.path
import pickle
from collections import namedtuple
from itertools import islice
from random import random, choice, sample
from typing import List, Dict, Set
from typing import TYPE_CHECKING

import numpy as np

import bsa.parasym
import bsa.scipy_sparse
import bsa.universal
import corona_matrix
from bsa.scipy_sparse import read_scipy_sparse
from common.social_circle import SocialCircle
from generation.circles_generator import PopulationData
from generation.connection_types import (
    Connect_To_All_types,
    ConnectionTypes,
    Geographic_Clustered_types,
    Random_Clustered_types,
)
from generation.matrix_consts import MatrixConsts, ConnectionTypeData
from generation.node import Node
from project_structure import OUTPUT_FOLDER

if TYPE_CHECKING:
    from parasymbolic_matrix.parasymbolic import ParasymbolicMatrix


class AgentConnections:
    def __init__(self):
        self.daily_connections: Set[int] = set()
        self.weekly_connections: Set[int] = set()


class ConnectionData:
    __slots__ = (
        "connected_ids_by_strength",
    )

    def __init__(self, agents):
        self.connected_ids_by_strength = {agent.index: {connection_type: AgentConnections() for connection_type
                                                        in ConnectionTypes} for agent in agents}

    def export(self, export_path, file_name="connection_data"):
        if not file_name.endswith(".pickle"):
            file_name += ".pickle"

        with open(os.path.join(export_path, file_name), "wb") as export_file:
            pickle.dump(self, export_file)

    @staticmethod
    def import_connection_data(import_file_path: str) -> "ConnectionData":
        with open(import_file_path, "rb") as import_file:
            connection_data = pickle.load(import_file)

        # pickle's version should be updated per application's version
        return connection_data


class MatrixData:
    __slots__ = ("matrix_type", "matrix", "depth")

    # import/export variables
    IMPORT_MATRIX_PATH = os.path.join(OUTPUT_FOLDER, "matrix_data")

    def __init__(self):
        self.matrix_type = None
        self.matrix: ParasymbolicMatrix = None
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
            raise FileNotFoundError("Couldn't open matrix data from {}".format(matrix_data_path))

    # todo make this work, using the parasymbolic matrix serialization.
    def export(self, export_path: str):
        if not export_path.endswith(self.matrix_type):
            export_path = export_path + '.' + self.matrix_type
        with open(export_path, 'wb') as f:
            bsa.universal.write(self.matrix, f)

    # todo Add support for other matrix types
    @staticmethod
    def import_matrix_data(import_file_path: str) -> "MatrixData":
        matrix_type = os.path.splitext(import_file_path)[1][1:]
        if matrix_type == 'parasymbolic':
            with open(import_file_path, 'rb') as f:
                matrix = bsa.parasym.read_parasym(f)
        matrix_data = MatrixData()
        matrix_data.matrix = matrix
        matrix_data.matrix_type = matrix_type
        matrix_data.depth = len(ConnectionTypes)  # This seems to essentialy be a constant.
        return matrix_data

    def get_scipy_sparse(self):
        """
        A wrapper for getting the scipy_sparse representation of the matrix.
        It doesn't change the current matrix, but creates a different one.
        :return: List[scipy.spars.lil_matrix] of #<depth> elements
        """
        b = bsa.parasym.write_parasym(self.matrix)
        b.seek(0)
        return bsa.scipy_sparse.read_scipy_sparse(b)


# todo right now only supports parasymbolic matrix. need to merge with corona matrix class import selector

MatrixAssignmentData = namedtuple('MatrixAssignmentData', ['depth', 'index', 'conns', 'v'])


class MatrixGenerator:
    """
    this module gets the circles and agents created in circles generator and creates a matrix and sub matrices with them.
    """

    def __init__(
            self, population_data: PopulationData, matrix_consts: MatrixConsts = MatrixConsts(),
    ):
        # initiate everything
        self.matrix_assignment_data = []
        self.logger = logging.getLogger("MatrixGenerator")
        self.matrix_data = MatrixData()
        self.connection_data = ConnectionData(population_data.agents)
        self.matrix_consts = matrix_consts
        self._unpack_population_data(population_data)
        self.size = len(self.agents)
        self.depth = len(ConnectionTypes)
        self.connection_types_data: Dict[ConnectionTypes, ConnectionTypeData] = {
            con_type: matrix_consts.get_connection_type_data(con_type) for con_type in ConnectionTypes}

        # TODO can we remove this? don't think we really support scipy matrices anymore
        CoronaMatrix = corona_matrix.get_corona_matrix_class(matrix_consts.use_parasymbolic_matrix)
        self.logger.info("Using CoronaMatrix of type {}".format(CoronaMatrix.__name__))
        self.matrix = CoronaMatrix(self.size, self.depth)

        # create all sub matrices
        # todo switch the depth logic, to get a connection type instead of int depth
        for current_depth, (con_type, con_type_data) in enumerate(self.connection_types_data.items()):
            if con_type in Connect_To_All_types:
                self._create_fully_connected_circles_matrix(
                    con_type_data, self.social_circles_by_connection_type[con_type], current_depth
                )
            elif con_type in Random_Clustered_types:
                self._create_scale_free_graph(
                    con_type_data, self.social_circles_by_connection_type[con_type], current_depth
                )
            elif con_type in Geographic_Clustered_types:
                self._create_randomly_connected_layer(
                    con_type_data, self.social_circles_by_connection_type[con_type], current_depth
                )

        with self.matrix.lock_rebuild():
            for depth, index, conns, v in self.matrix_assignment_data:
                self.matrix[depth, index, conns] = v

        # current patch since matrix is un-serializable
        self.matrix_data.matrix_type = "parasymbolic"
        self.matrix_data.matrix = self.matrix
        self.matrix_data.depth = self.depth

    def _unpack_population_data(self, population_data):
        self.agents = population_data.agents
        self.social_circles_by_connection_type = population_data.social_circles_by_connection_type
        self.geographic_circles = population_data.geographic_circles
        self.geographic_circle_by_agent_index = population_data.geographic_circle_by_agent_index
        self.social_circles_by_agent_index = population_data.social_circles_by_agent_index

    def _add_layer(self, con_type_data: ConnectionTypeData, connections: List[List[int]], depth: int):
        # insert all connections to matrix
        # we need to remember the strengths so the connection will be symmetric
        known_strengths = {}
        for agent, conns in zip(self.agents, connections):
            if len(conns) == 0:
                continue

            conns = np.array(conns)
            conns.sort()

            strengthes = con_type_data.get_strengths(len(conns))

            # check if some strengths were determined earlier
            for index, conn in enumerate(conns):
                known_strength = known_strengths.get((conn, agent.index), None)
                if known_strength != None:
                    strengthes[index] = known_strength
                    del known_strengths[(conn, agent.index)]
                else:
                    # connection is new. store the strength for future use
                    known_strengths[(agent.index, conn)] = strengthes[index]

                if strengthes[index] == con_type_data.connection_strength:
                    self.connection_data.connected_ids_by_strength[agent.index][
                        con_type_data.connection_type].daily_connections.add(conn)
                else:
                    self.connection_data.connected_ids_by_strength[agent.index][
                        con_type_data.connection_type].weekly_connections.add(conn)

            v = np.full_like(conns, strengthes, dtype=np.float32)
            self.matrix_assignment_data.append(MatrixAssignmentData(depth, agent.index, conns, v.copy()))

    def _create_fully_connected_circles_matrix(self, con_type_data: ConnectionTypeData, circles: List[SocialCircle],
                                               depth):
        connection_strength = con_type_data.connection_strength
        if connection_strength == 0:
            return

        for circle in circles:
            if circle.agent_count <= 1:
                # An empty circle (shouldn't happen) or a single-agent circle (there isn't any meaning to the
                # connection strength between an agent to itself)
                continue
            ids = np.array([a.index for a in circle.agents])

            vals = np.full_like(ids, connection_strength, dtype=np.float32)
            for i, agent in enumerate(circle.agents):
                temp = vals[i]
                vals[i] = 0
                self.matrix_assignment_data.append(MatrixAssignmentData(depth, int(agent.index), ids, vals.copy()))
                vals[i] = temp
                self.connection_data.connected_ids_by_strength[agent.index][
                    con_type_data.connection_type].daily_connections.update(set(ids))

    def _create_scale_free_graph(self, con_type_data: ConnectionTypeData, circles: List[SocialCircle], depth):
        # the new connections will be saved here
        connections = [[] for _ in self.agents]
        # gets data from matrix consts
        connection_strength = con_type_data.connection_strength
        if connection_strength == 0:
            return

        # adding all super small circles, into one circle, and randomly create connections inside it
        super_small_circles_combined = SocialCircle(con_type_data.connection_type)

        for circle in circles:
            agents = circle.agents
            indexes = [agent.index for agent in agents]
            nodes: List[Node] = [Node(index) for index in indexes]

            # the number of nodes. writes it for simplicity
            n = len(indexes)
            connections_amounts = con_type_data.get_scale_free_connections_amount(shape=n)

            # saves the already-inserted nodes
            np.random.shuffle(nodes)
            initial_con_amount = math.ceil(con_type_data.total_connections_amount) + 1

            # checks, if the circle is too small for any algorithm. if so adds to super small circle
            if n < initial_con_amount:
                super_small_circles_combined.add_many(circle.agents)
                continue

            # checks, if the circle is too small for normal clustering
            if n < self.matrix_consts.clustering_switching_point:
                self._randomly_connect_single_circle(circle, connections, con_type_data.total_connections_amount)
                continue

            connected_nodes = set()

            # manually generate the minimum required connections
            for i in range(initial_con_amount):
                other_nodes = nodes[0:initial_con_amount]
                other_nodes.pop(i)
                nodes[i].add_connections(other_nodes)
                connected_nodes.add(nodes[i])

            # add the rest of the nodes, one at a time
            for node, num_connections in zip(islice(nodes, initial_con_amount, None), connections_amounts):
                # selects the first node to attach to randomly
                first_connection = choice(list(connected_nodes))
                # fill connections other than first_connection
                while len(node.connected) < num_connections - 1:
                    if random() < con_type_data.triad_p:
                        # close the triad with a node from first_connection's connections
                        possible_nodes = first_connection.connected
                    else:
                        # connect with a node, which is not first connection
                        possible_nodes = connected_nodes

                    # prevent connecting a connected node
                    if len(possible_nodes) - len(node.connected) > 100:  # TODO 100 is made up. should be optimized
                        possible_list = list(possible_nodes)
                        random_friend = choice(possible_list)
                        while random_friend in node.connected:
                            random_friend = choice(possible_list)
                    else:
                        possible_nodes = possible_nodes.difference(node.connected).difference({first_connection})
                        random_friend = choice(list(possible_nodes))
                    Node.connect(random_friend, node)
                # connect to bff here to prevent self-selection in bff's friends
                Node.connect(first_connection, node)
                connected_nodes.add(node)

            for connected_node in nodes:
                connections[connected_node.index].extend([other_node.index for other_node in connected_node.connected])

        # adding connections between all super small circles
        self._randomly_connect_single_circle(super_small_circles_combined, connections,
                                             con_type_data.total_connections_amount)

        # insert connections to matrix
        self._add_layer(con_type_data, connections, depth)

    def _create_randomly_connected_layer(self, con_type_data: ConnectionTypeData,
                                         circles: List[SocialCircle], depth):
        # the new connections will be saved here
        connections = [[] for _ in self.agents]
        # gets data from matrix consts
        connection_strength = con_type_data.connection_strength
        if connection_strength == 0:
            return

        for circle in circles:
            self._randomly_connect_single_circle(circle, connections, con_type_data.total_connections_amount)

        # insert all connections to matrix
        self._add_layer(con_type_data, connections, depth)

    def _randomly_connect_single_circle(self, circle: SocialCircle, connections: List[List], scale_factor: float):
        """
        creates circle's connections, and adds them to a given connections list
        connections amount will be generated from an exponential distribution
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
            conns = sample(list(agent_id_pool), rc)
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

    def export_matrix_data(self, export_dir=OUTPUT_FOLDER, export_filename='matrix.bsa'):
        self.matrix_data.export(os.path.join(export_dir, export_filename))
