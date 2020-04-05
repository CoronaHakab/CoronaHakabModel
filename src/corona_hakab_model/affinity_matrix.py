import logging
import math
from random import sample, shuffle
from typing import Iterable, List

import numpy as np
from agent import Agent, TrackingCircle
from node import Node
from consts import Consts
from itertools import islice

use_parasymbolic_matrix = True
if use_parasymbolic_matrix:
    from sparse_matrix import ParasymbolicMatrix as CoronaMatrix
else:
    from scipy_matrix import ScipyMatrix as CoronaMatrix


class AffinityMatrix:
    """
    This class builds and maintains the sparse affinity matrix W which describes the social connections
    (the social circles).
    W is NxN, where N is the total population size.
    If W(i,j) is large, this means that node (person) i is socially close to node j.
    Thus, nodes i and j can easily infect one another.
    Naturally, W is symmetric.
    """

    def __init__(self, agents: Iterable[Agent], consts: Consts):
        self.consts = consts
        self.size = consts.population_size
        assert len(agents) == consts.population_size, "Size of population doesn't match agent list size!"
        self.logger = logging.getLogger("simulation")

        self.logger.info("Building new AffinityMatrix")

        self.circular_matrix_types = {}
        for i, cm in enumerate(self.consts.circular_matrices):
            self.circular_matrix_types[cm.name] = (i, cm)

        self.non_circular_matrix_types = {}
        for j, ncm in enumerate(self.consts.non_circular_matrices, len(self.circular_matrix_types)):
            self.non_circular_matrix_types[ncm.name] = (j, ncm)

        self.clustered_matrix_types = {}
        for k, clm in enumerate(self.consts.clustered_matrices,
                                len(self.circular_matrix_types) + len(self.non_circular_matrix_types)):
            self.clustered_matrix_types[clm.name] = (k, clm)

        self.depth = k + 1
        self.inner = CoronaMatrix(self.size, self.depth)

        self.agents = agents

        self.logger.info("Building circular connections matrices")

        with self.inner.lock_rebuild():
            # all circular matrices. keeping as a tuple of matrix and type (i.e, home, work, school and so)
            for i, cm in self.circular_matrix_types.values():
                self.build_cm(i, cm)

            self.logger.info("Building non circular connections matrices")
            # all non-circular matrices. keeping as a tuple of matrix and type (i.e, home, work, school and so)
            for i, ncm in self.non_circular_matrix_types.values():
                self.build_ncm(i, ncm)

            self.logger.info("Building clustered connections matrices")
            # all clustered matrices. keeping as a tuple of matrix and type (i.e, home, work, school and so)
            for i, clm in self.clustered_matrix_types.values():
                self.build_clm(i, clm)

            self.logger.info("summing all matrices")

        self.normalize_factor = None
        self.total_contagious_probability = None

        self.normalize()

    def normalize(self):
        """
        this function should normalize the weights within W to represent the infection rate.
        As r0=bd, where b is number of daily infections per person
        """
        self.logger.info(f"normalizing matrix")
        if self.normalize_factor is None:
            # updates r0 to fit the contagious length and ratio.
            states_time = self.consts.average_time_in_each_state()
            total_contagious_probability = 0
            for state, time_in_state in states_time.items():
                total_contagious_probability += time_in_state * state.contagiousness
            beta = self.consts.r0 / total_contagious_probability

            # saves this for the effective r0 graph
            self.total_contagious_probability = total_contagious_probability

            # this factor should be calculated once when the matrix is full, and be left un-changed for the rest of the run.
            self.normalize_factor = (beta * self.size) / (self.inner.total())

        self.inner *= self.normalize_factor  # now each entry in W is such that bd=R0

    def change_connections_policy(self, types_of_connections_to_use: Iterable[str]):
        self.logger.info(f"changing policy. keeping all matrices of types: {types_of_connections_to_use}")
        factors = np.zeros(self.depth, dtype=np.float32)
        for t in types_of_connections_to_use:
            ind, _ = self.circular_matrix_types.get(t) or self.non_circular_matrix_types.get(t) or self.clustered_matrix_types[t]
            factors[ind] = 1
        self.inner.set_factors(factors)
        self.normalize()

    def build_cm(self, depth, cm):
        circles_size_rolls = iter(
            cm.circle_size_probability.rvs(size=math.ceil(len(self.agents) / cm.circle_size_probability.mean() + 10))
        )

        circles: List[TrackingCircle] = []
        agent_queue = list(self.agents)
        shuffle(agent_queue)
        while len(agent_queue) > 0:
            # todo does this have to be a tracking circle? or even a circle at all?
            circle = TrackingCircle()
            try:
                current_circle_size = circles_size_rolls.__next__()
            except StopIteration:
                current_circle_size = cm.circle_size_probability.rvs()

            # adding agents to the current circle
            agents = agent_queue[-current_circle_size:]
            del agent_queue[-current_circle_size:]

            circle.add_many(agents)
            # todo possibly add this circle to the agent circles list
            circles.append(circle)

        for circle in circles:
            ids = np.array([a.index for a in circle.agents])
            vals = np.full_like(ids, cm.connection_strength, dtype=np.float32)
            for i, row in enumerate(ids):
                temp = vals[i]
                vals[i] = 0
                self.inner[depth, int(row), ids] = vals
                vals[i] = temp

    def build_clm(self, depth, clm):

        # extract data from clm
        agents = clm.agents
        if agents is None:
            agents = self.agents
        mean_connections_amount = clm.mean_connections_amount
        connection_strength = clm.connection_strength

        # the new connections will be saved here
        connections = [[] for _ in self.agents]

        indexes = [agent.index for agent in agents]
        # list of nodes. each contains his id and a list of neighbers
        nodes: List[Node] = [Node(index) for index in indexes]
        # the number of edges that will be made with each addition of node to the graph
        m = mean_connections_amount // 2
        # the number of nodes. writes it for simplicity
        n = len(indexes)
        # pre-generates all rolls for efficiency
        rolls = iter(np.random.random(n * (1 + m)))
        # saves the already-instered nodes
        inserted_nodes = []

        np.random.shuffle(nodes)

        # manually generate the first m + 1 connections
        for i in range(m + 1):
            other_nodes = nodes[0: m + 1]
            other_nodes.pop(i)
            nodes[i].add_connections(other_nodes)
            inserted_nodes.append(nodes[i])

            # add the newly made connections to the connections list. note that this is one directional,
            # but the other direction will be added when adding the other's connections
            connections[nodes[i].index].extend([node.index for node in other_nodes])

        # add the rest of the nodes, one at a time
        for node in islice(nodes, m+1, None):
            # randomly select the first node to connect. pops him so that he won't be choosen again
            rand_node = inserted_nodes.pop(math.floor(rolls.__next__() * len(inserted_nodes)))

            # add connection to connections list
            connections[node.index].append(rand_node.index)
            connections[rand_node.index].append(node.index)

            # todo change this to use p, and not only p = 1
            # randomly choose the rest of the connections from rand_node connections.
            nodes_to_return = []
            for _ in range(m - 1):
                # randomly choose a node from rand_node connections
                new_rand = rand_node.pop_random(rolls.__next__())
                nodes_to_return.append(new_rand)
                Node.connect(node, new_rand)

                # add connection to connections list
                connections[node.index].append(new_rand.index)
                connections[new_rand.index].append(node.index)

            # connect current node with rand node. note that this only happens here to not pick yourself later on
            Node.connect(node, rand_node)
            # return the popped nodes back to rand_node connections, and rand node back to already inserted list
            rand_node.add_connections(nodes_to_return)
            inserted_nodes.append(rand_node)
            inserted_nodes.append(node)

        # insert all connections to matrix
        for agent, conns in zip(self.agents, connections):
            conns = np.array(conns)
            conns.sort()
            v = np.full_like(conns, connection_strength, dtype=np.float32)
            self.inner[depth, agent.index, conns] = v


    def build_ncm(self, depth, ncm):
        # an iterator representing the rolled amount of connections per agent
        # todo note that alpha is beeing reduced by 0.5, because later on it is getting math.ceil. it will not change the mean but will change the distribution

        remaining_contacts = np.ceil(np.random.exponential(ncm.scale_factor - 0.5, len(self.agents))).astype(int)
        connections = [[] for _ in self.agents]

        agent_id_pool = set(range(len(self.agents)))
        print("d",depth)

        # while there are still connections left to make
        print("drawing")
        while len(agent_id_pool) >= 2:
            current_agent_id = agent_id_pool.pop()

            # todo i don't like this min, it indicates an issue
            rc = min(remaining_contacts[current_agent_id], len(agent_id_pool))
            conns = np.array(sample(agent_id_pool, rc))
            connections[current_agent_id].extend(conns)
            for other_agent_id in conns:
                connections[other_agent_id].append(current_agent_id)
            remaining_contacts[conns] -= 1

            to_remove = set(conns[remaining_contacts[conns] == 0])
            #assert to_remove <= agent_id_pool

            agent_id_pool.difference_update(to_remove)
        print("setting")
        for agent, conns in zip(self.agents, connections):
            conns = np.array(conns)
            conns.sort()
            v = np.full_like(conns, ncm.connection_strength, dtype=np.float32)
            self.inner[depth, agent.index, conns] = v
