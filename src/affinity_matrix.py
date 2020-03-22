from itertools import product
from math import log
from random import shuffle

import numpy as np
from scipy.sparse import lil_matrix
from agent import Agent, Circle
import random as rnd
import corona_stats, social_stats

m_type = lil_matrix


class AffinityMAtrix:
    """
    This class builds and maintains the sparse affinity matrix W which describes the social connections
    (the social circles).
    W is NxN, where N is the total population size.
    If W(i,j) is large, this means that node (person) i is socially close to node j.
    Thus, nodes i and j can easly infect one another.
    Naturally, W is symetric.
    """

    def __init__(self, size):
        self.size = size  # population size
        self.matrix = m_type((size, size), dtype=np.float16)

        self.agents = self.generate_agents()

        self.m_families = self._create_intra_family_connections()
        self.m_work = self._create_intra_workplace_connections()
        self.m_random = self._create_random_connectivity()

        self.matrix = self.m_families + self.m_work + self.m_random

        self.normalize()

    def generate_agents(self):
        agents = [Agent(id) for id in range(self.size)]
        return agents

    def _create_intra_family_connections(self):
        """
        here need to build random buckets of size N/self.averageFamilySize
        and add nodes to a NxN sparce amtrix W_famillies describing the connections within each family.
        If for example, nodes 1 till 5 are a family, we need to build connections between each and
        every memeber of this family. The value of each edge should be heigh, representing a
        high chance of passing the virus, since the family members stay a long time together.
        In the example of nodes 1 till 5 buing a family, in Matlab this would be: W_families[1:5,1:5]=p
        where p is the intra family infection probability.
        Late on, if, for example, a policy of house containments takes place without the members of the family
        taking measures to separate from each other, then this value p can be replaced by something even larger.
        """
        # as a beggining, i am making all families the same size, later we will change it to be more sophisticated
        matrix = m_type((self.size, self.size), dtype=np.float32)

        # creating all families, and assigning each agent to a family, and counterwise
        agents_without_home = list(range(self.size))
        shuffle(agents_without_home)
        families = []
        for _ in range(self.size // social_stats.avarage_family_size):
            new_family = Circle("home")
            for _ in range(social_stats.avarage_family_size):
                chosen_agent = self.agents[agents_without_home.pop()]
                chosen_agent.add_home(new_family)
                new_family.add_agent(chosen_agent)
            families.append(new_family)
        self.families = families

        # adding the remaining people to a family (if size % average_family_size != 0)
        if len(agents_without_home) > 0:
            new_family = Circle("home")
            for agent_index in agents_without_home:
                chosen_agent = self.agents[agent_index]
                chosen_agent.add_home(new_family)
                new_family.add_agent(chosen_agent)
            families.append(new_family)

        for home in families:
            ids = np.array([a.ID for a in home.agents])
            xs, ys = np.meshgrid(ids, ids)
            xs = xs.reshape(-1)
            ys = ys.reshape(-1)
            matrix[xs, ys] = social_stats.family_strength
        ids = np.arange(self.size)
        matrix[ids, ids] = 0

        return matrix

    def _create_intra_workplace_connections(self):
        """
        Similar to build the family connections we here build the working place connections
        divide the population which goes to work (say 0.4N) into buckets of size that correspond
        to work place size.
        Within the nodes of each bucket (i.e. each work place), make some random connections according
        to the number of close colleagues each person might have.

        :return: lil_matrix n*n
        """
        # note: a bug in numpy casting will cause a crash on array inset with float16 arrays, we should use float32
        matrix = m_type((self.size, self.size), dtype=np.float32)

        # creating all families, and assigning each agent to a family, and counterwise
        agents_without_work = list(range(self.size))
        shuffle(agents_without_work)
        works = []
        for _ in range(self.size // social_stats.avarage_work_size):  # todo add last work
            new_work = Circle("work")
            for _ in range(social_stats.avarage_work_size):
                chosen_agent_ind = agents_without_work.pop()

                chosen_agent = self.agents[chosen_agent_ind]
                chosen_agent.add_work(new_work)
                new_work.add_agent(chosen_agent)
            works.append(new_work)
        self.works = works

        # adding the remaining people to a work (if size % average_work_size != 0)
        if len(agents_without_work) > 0:
            new_work = Circle("work")
            for agent_index in agents_without_work:
                chosen_agent = self.agents[agent_index]
                chosen_agent.add_work(new_work)
                new_work.add_agent(chosen_agent)
            works.append(new_work)

        # updating the matrix using the works
        for work in works:
            ids = np.array([a.ID for a in work.agents])
            xs, ys = np.meshgrid(ids, ids)
            xs = xs.reshape(-1)
            ys = ys.reshape(-1)
            matrix[xs, ys] = social_stats.work_strength
        ids = np.arange(self.size)
        matrix[ids, ids] = 0
        return matrix

    def _create_random_connectivity(self):
        """
        plug here random connection, super spreaders, whatever. We can also adjust the number of daily connections
        b or beta in the literature) by adding this random edges
        :return: lil_matrix n*n
        """
        matrix = m_type((self.size, self.size), dtype=np.float32)
        amount_of_connections = social_stats.average_amount_of_strangers

        stranger_ids = np.random.randint(0, self.size - 1, self.size * amount_of_connections)
        ids = np.arange(self.size).repeat(amount_of_connections)

        matrix[ids, stranger_ids] = social_stats.stranger_strength
        return matrix

    def normalize(self):
        """
        this funciton should normalize the weights within W to represent the infection rate.
        As r0=bd, where b is number of daily infections per person
        """
        r0 = corona_stats.r0
        non_zero_elements = self.matrix.count_nonzero()

        b = non_zero_elements / self.size  # average number of connections per person per day
        d = r0 / corona_stats.average_infection_length / b  # avarage probability for infection in each meeting as should be
        average_edge_weight_in_matrix = self.matrix.sum() / non_zero_elements  # avarage probability for infection in each meeting in current matrix
        self.matrix = self.matrix * d / average_edge_weight_in_matrix  # (alpha = d / average_edge_weight_in_matrix) now each entry in W is such that bd=R0
        social_stats.family_strength_not_workers = social_stats.family_strength_not_workers * d / average_edge_weight_in_matrix

        # switching from probability to ln(1-p):
        non_zero_keys = self.matrix.nonzero()
        self.matrix[non_zero_keys] = np.log(1 - self.matrix[non_zero_keys])


def dot(self, v):
    """
    performs dot operation between this matrix and v
    :param v: with the size of self.size

    :return: matrix*v
    """

    return self.matrix.dot(v)


def zero_column(col_id):
    """
    Turn the chosen column to zeroes.
    """
    pass


def add_to_column(col_id, col):
    """
    Add the given column to the chosen column
    """
    pass
