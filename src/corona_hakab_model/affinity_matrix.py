import logging
import math
from typing import Dict, List, Sequence

import numpy as np
from agent import Agent, TrackingCircle
from node import Node
from scipy.sparse import lil_matrix, load_npz, save_npz
from scipy.stats import rv_discrete

m_type = lil_matrix


class AffinityMatrix:
    """
    This class builds and maintains the sparse affinity matrix W which describes the social connections
    (the social circles).
    W is NxN, where N is the total population size.
    If W(i,j) is large, this means that node (person) i is socially close to node j.
    Thus, nodes i and j can easily infect one another.
    Naturally, W is symmetric.
    """

    def __init__(self, manager, input_matrix_path: str = None, output_matrix_path: str = None):
        self.consts = manager.consts
        self.size = len(manager.agents)  # population size
        self.logger = logging.getLogger("simulation")

        self.manager = manager
        if input_matrix_path and m_type == lil_matrix:
            self.logger.info(f"Loading matrix from file: {input_matrix_path}")
            try:
                with open(input_matrix_path, "rb") as f_matrix:
                    self.matrix = load_npz(f_matrix)
            except FileNotFoundError as e:
                self.logger.error(f"File {input_matrix_path} not found!")
                raise e
            self.logger.info("Matrix loaded succesfully")
            return

        self.logger.info("Building new AffinityMatrix")
        self.matrix = m_type((self.size, self.size), dtype=np.float32)

        self.agents = self.manager.agents

        self.logger.info("Building circular connections matrices")

        # all circular matrices. keeping as a tuple of matrix and type (i.e, home, work, school and so)
        self.circular_matrices = [
            (
                self.circular_matrix_generation(
                    self.agents, matrix.circle_size_probability, matrix.connection_strength
                ).tocsr(),
                matrix.type,
            )
            for matrix in self.consts.circular_matrices
        ]

        self.logger.info("Building non circular connections matrices")
        # all non-circular matrices. keeping as a tuple of matrix and type (i.e, home, work, school and so)
        self.non_circular_matrices = [
            (
                self.non_circular_matrix_generation(
                    self.agents, matrix.scale_factor, matrix.connection_strength
                ).tocsr(),
                matrix.type,
            )
            for matrix in self.consts.non_circular_matrices
        ]

        self.logger.info("Building clustered connections matrices")
        # all clustered matrices. keeping as a tuple of matrix and type
        self.clustered_matrices = [
            (
                self.clustered_matrix_generation(
                    self.agents, matrix.mean_connections_amount, matrix.connection_strength
                ).tocsr(),
                matrix.type,
            )
            for matrix in self.consts.clustered_matrices
        ]

        # saves all sub matrices under one variable
        self.sub_matrices = self.circular_matrices
        self.sub_matrices.extend(self.non_circular_matrices)
        self.sub_matrices.extend(self.clustered_matrices)

        self.logger.info("summing all matrices")
        self.matrix = sum(matrix[0] for matrix in self.sub_matrices)

        self.factor = None
        self.normalize()

        if output_matrix_path and m_type == lil_matrix:
            self.logger.info(f"Saving AffinityMatrix internal matrix to {output_matrix_path}")
            try:
                with open(output_matrix_path, "wb") as f_matrix:
                    save_npz(f_matrix, self.matrix)
            except FileNotFoundError:
                self.logger.error(f"Path {output_matrix_path} is invalid!")
            self.logger.info("Matrix saved successfully!")

    def normalize(self):
        """
        this function should normalize the weights within W to represent the infection rate.
        As r0=bd, where b is number of daily infections per person
        """
        self.logger.info(f"normalizing matrix")
        if self.factor is None:
            # updates r0 to fit the contagious length and ratio.
            states_time = self.consts.average_time_in_each_state()
            total_contagious_probability = 0
            for state, time_in_state in states_time.items():
                total_contagious_probability += time_in_state * state.contagiousness
            beta = self.consts.r0 / total_contagious_probability

            # saves this for the effective r0 graph
            self.total_contagious_probability = total_contagious_probability

            # this factor should be calculated once when the matrix is full, and be left un-changed for the rest of the run.
            self.factor = (beta * self.size) / (self.matrix.sum())

        self.matrix = self.matrix * self.factor  # now each entry in W is such that bd=R0

        # switching from probability to ln(1-p):
        non_zero_keys = self.matrix.nonzero()
        self.matrix[non_zero_keys] = np.log(1 - self.matrix[non_zero_keys])

    def change_connections_policy(self, types_of_connections_to_use: Sequence[str]):
        self.logger.info(f"changing policy. keeping all matrices of types: {types_of_connections_to_use}")
        self.matrix = sum(matrix[0] for matrix in self.sub_matrices if matrix[1] in types_of_connections_to_use)
        self.normalize()

    def non_circular_matrix_generation(self, agents_to_use, scale_factor: float, connection_strength):
        """
        creates a matrix of non-circular connections. the amount of connections per agent goes by exponential distribution. each connection will be with the given connection strength
        :param agents_to_use: a list of agents to add to this matrix. doesnt have to be all agents
        :param scale_factor: the scale factor of the exponential distribution (1/alpha)
        :param connection_strength: the connection strength that will be used to describe those connections
        :return: a lil matrix representing the connections
        """
        # the lil_matrix to be returned
        matrix = m_type((self.size, self.size), dtype=np.float32)

        # calculates it only once for effifiency
        amount_of_agents = len(agents_to_use)

        # an iterator representing the rolled amount of connections per agent
        # todo note that alpha is beeing reduced by 0.5, because later on it is getting math.ceil. it will not change the mean but will change the distribution
        amount_of_connections = np.ceil(np.random.exponential(scale_factor - 0.5, amount_of_agents)).astype(int)

        # dict of agents to amount of remaining connections to make for this agent
        remaining_contacts: Dict[Agent, int] = {
            agent: amount for (agent, amount) in zip(agents_to_use, amount_of_connections)
        }

        # will be used as a stopping sign. math.ceil because np ceil is returning floats, and summing a lot of floats has a cumulative error
        connections_sum = sum(remaining_contacts.values())

        # a list of indexes of agents which still lack connections. used for efficiency
        available_agents = list(agents_to_use)

        # pre-rolling all rolls for efficiency. for each connection, the 2nd agent will be rolled using this
        # todo make sure the later-used % operator doesn't harm the randomness
        rolls = iter(np.random.randint(0, amount_of_agents, connections_sum // 2 + 1))

        # this structure will be used to save all connections, and later on insert them all at once. used for efficiency
        connections_tuples = np.zeros((2, connections_sum), dtype=np.int)
        connections_cnt = 0

        # while there are still connections left to make
        while len(available_agents) >= 2 and connections_sum >= 2:

            current_agent = available_agents.pop()

            # temp holder for used agents, so that the same connection wont be made twice
            temp_agents_holder = []

            # creating all of first's connections
            for _ in range(remaining_contacts[current_agent]):
                if connections_sum <= 1 or len(available_agents) <= 1:
                    break

                # choosing 2nd agent for the connection
                # todo make sure the % operator doesn't harm the randomness too much
                next_roll = rolls.__next__() % len(available_agents)
                second_agent = available_agents[next_roll]

                # adding the newly made connection to the to-be-added connections structure
                connections_tuples[0, connections_cnt] = current_agent.index
                connections_tuples[1, connections_cnt] = second_agent.index
                connections_cnt += 1
                connections_tuples[0, connections_cnt] = second_agent.index
                connections_tuples[1, connections_cnt] = current_agent.index
                connections_cnt += 1

                # note that there is no reason to update firs's remaining connections for efficiency
                remaining_contacts[second_agent] -= 1
                connections_sum -= 2

                if remaining_contacts[second_agent] <= 0:
                    del available_agents[next_roll]
                else:
                    temp_agents_holder.append(available_agents.pop(next_roll))
            # returning all the agents back from the temp place holder
            available_agents.extend(temp_agents_holder)

        # filling the matrix. sometimes there still remains 1 un-filled connection, and it is left as 0,0 connection, so reset 0,0
        matrix[connections_tuples[0], connections_tuples[1]] = connection_strength
        matrix[0, 0] = 0

        return matrix

    def circular_matrix_generation(
        self, agents: List[Agent], circle_size_probability: rv_discrete, connection_strength
    ):
        """
        this method will create a matrix of circular connections given a list of agents, an rv_discrete representing the size of the circle probabilty, and the connection strength
        :param agents: list of all agents
        :param circle_size_probability: representing the size probability
        :param connection_strength: the wanted connection strength
        :return: lil_matrix with the wanted connections
        """
        matrix = m_type((self.size, self.size), dtype=np.float32)

        # pre-rolling all the sized of the circles for efficiency causes.
        circles_size_rolls = iter(
            circle_size_probability.rvs(size=math.ceil(len(agents) / circle_size_probability.mean() + 10))
        )

        # here all the circles will be saved
        circles: List[TrackingCircle] = []

        # using a copy to not change the main agents list
        agents_copy = list(agents)
        np.random.shuffle(agents_copy)

        # loop creating all circles. each run creates one circle.
        while len(agents_copy) > 0:

            current_circle = TrackingCircle()

            # choosing current circle size
            current_circle_size = 0
            try:
                current_circle_size = circles_size_rolls.__next__()
            except StopIteration:
                current_circle_size = circle_size_probability.rvs()

            # adding agents to the current circle
            for _ in range(current_circle_size):
                if len(agents_copy) <= 0:
                    break
                choosen_agent = agents_copy.pop()
                current_circle.add_agent(choosen_agent)
                # todo possibly add this circle to the agent circles list
            circles.append(current_circle)

        for circle in circles:
            ids = np.array([a.index for a in circle.agents])
            xs, ys = np.meshgrid(ids, ids)
            xs = xs.reshape(-1)
            ys = ys.reshape(-1)
            matrix[xs, ys] = connection_strength

        # note that the previous loop also creates a connection between each agent and himself. this part removes it
        ids = [agent.index for agent in agents]
        matrix[ids, ids] = 0
        return matrix

    def clustered_matrix_generation(
        self, agents: List[Agent], mean_connections_amount: int, connection_strength=1, p: float = 1
    ):
        """
        returns a matrix of clustered connections.
        :param agents: agents to use in this connection
        :param mean_connections_amount: the average amount of connections wanted (1 / alpha in exponential distribution)
        :param connection_strength: the strength of each connection
        :param p: another part of the model. currently un-used, possible to add later or
        :return:
        """
        indexes = [agent.index for agent in agents]
        # matrix to fill and return
        matrix = lil_matrix((self.size, self.size), dtype=np.float32)
        # list of nodes. each contains his id and a list of neighbers
        nodes: List[Node] = [Node(index) for index in indexes]
        # the number of edges that will be made with each addition of node to the graph
        m = mean_connections_amount // 2
        # the number of nodes. writes it for simplicity
        n = len(indexes)
        # saves the connections to add, so that filling the matrix will be done only 1 time (more efficient)
        connections = np.zeros((2, len(indexes) * m), dtype=int)
        connections_cnt = 0
        # pre-generates all rolls for efficiency
        rolls = iter(np.random.random(n * (1 + m)))
        # saves the already-instered nodes
        inserted_nodes = []

        np.random.shuffle(nodes)

        # manually generate the first m + 1 connections
        for i in range(m + 1):
            other_nodes = nodes[0 : m + 1]
            other_nodes.pop(i)
            nodes[i].add_connections(other_nodes)
            inserted_nodes.append(nodes[i])

            # add the newly made connections to the connections list
            connections[0, connections_cnt : connections_cnt + len(other_nodes)] = nodes[i].index
            connections[1, connections_cnt : connections_cnt + len(other_nodes)] = np.fromiter(
                [node.index for node in other_nodes], int
            )
            connections_cnt += len(other_nodes)

        # add the rest of the nodes, one at a time
        for node in nodes[m + 1 :]:
            # randomly select the first node to connect. pops him so that he won't be choosen again
            rand_node = inserted_nodes.pop(math.floor(rolls.__next__() * len(inserted_nodes)))

            # add connection to connections list
            connections[0, connections_cnt] = node.index
            connections[1, connections_cnt] = rand_node.index
            connections_cnt += 1

            # todo change this to use p, and not only p = 1
            # randomly choose the rest of the connections from rand_node connections.
            nodes_to_return = []
            for _ in range(m - 1):
                # randomly choose a node from rand_node connections
                new_rand = rand_node.pop_random(rolls.__next__())
                nodes_to_return.append(new_rand)
                Node.connect(node, new_rand)

                # add connection to connections list
                connections[0, connections_cnt] = node.index
                connections[1, connections_cnt] = new_rand.index
                connections_cnt += 1

            # connect current node with rand node. note that this only happens here to not pick yourself later on
            Node.connect(node, rand_node)
            # return the popped nodes back to rand_node connections, and rand node back to already inserted list
            rand_node.add_connections(nodes_to_return)
            inserted_nodes.append(rand_node)
            inserted_nodes.append(node)

        # insert all connections to matrix
        matrix[connections[0], connections[1]] = connection_strength
        matrix[connections[1], connections[0]] = connection_strength
        return matrix
