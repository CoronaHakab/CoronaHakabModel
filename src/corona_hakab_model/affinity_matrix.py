import logging
from random import shuffle

import numpy as np
from agent import TrackingCircle
from scipy.sparse import lil_matrix

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

    def __init__(self, manager):
        self.consts = manager.consts
        self.size = len(manager.agents)  # population size

        self.manager = manager

        self.matrix = m_type((self.size, self.size), dtype=np.float32)
        self.logger = logging.getLogger("simulation")
        self.logger.info("Building new AffinityMatrix")
        self.agents = self.manager.agents

        self.m_families = self._create_intra_family_connections()
        self.m_work = self._create_intra_workplace_connections()
        self.m_random = self._create_random_connectivity()

        # switches all matrices to csr, for efficiency later on (in home isolation calculations)
        self.m_families = self.m_families.tocsr()
        self.m_work = self.m_work.tocsr()
        self.m_random = self.m_random.tocsr()

        self.matrix = self.m_families + self.m_work + self.m_random

        self.factor = None
        self.normalize()

    def _create_intra_family_connections(self):
        """
        here need to build random buckets of size N/self.averageFamilySize
        and add nodes to a NxN sparse matrix W_families describing the connections within each family.
        If for example, nodes 1 till 5 are a family, we need to build connections between each and
        every member of this family. The value of each edge should be high, representing a
        high chance of passing the virus, since the family members stay a long time together.
        In the example of nodes 1 till 5 are a family, in Matlab this would be: W_families[1:5,1:5]=p
        where p is the intra family infection probability.
        Late on, if, for example, a policy of house isolation takes place without the members of the family
        taking measures to separate from each other, then this value p can be replaced by something even larger.
        """

        self.logger.info(f"Create intra family connections")
        # as a beginning, I am making all families the same size, later we will change it to be more sophisticated

        matrix = m_type((self.size, self.size), dtype=np.float32)

        # creating all families, and assigning each agent to a family, and counter-wise
        agents_without_home = list(range(self.size))
        shuffle(agents_without_home)
        families = []
        num_of_families = self.size // self.consts.average_family_size
        for i in range(num_of_families):
            if i % (num_of_families // 100) == 0:
                self.logger.info(f"Creating family {i}/{num_of_families}")
            new_family = TrackingCircle()
            for _ in range(self.consts.average_family_size):
                chosen_agent = self.agents[agents_without_home.pop()]
                chosen_agent.add_home(new_family)
                new_family.add_agent(chosen_agent)
            families.append(new_family)
        self.families = families

        # adding the remaining people to a family (if size % average_family_size != 0)
        if len(agents_without_home) > 0:
            self.logger.info("adding remaining agents to families")
            new_family = TrackingCircle()
            for agent_index in agents_without_home:
                chosen_agent = self.agents[agent_index]
                chosen_agent.add_home(new_family)
                new_family.add_agent(chosen_agent)
            families.append(new_family)

        # setting the connection strength between the agents in the matrix
        for family in families:
            # extracting indexes of the agents in the family, which will serve as coordinates in the meshg rid
            ids = np.array([a.index for a in family.agents])
            xs, ys = np.meshgrid(ids, ids)
            xs = xs.reshape(-1)
            ys = ys.reshape(-1)
            matrix[xs, ys] = self.consts.family_strength
        # setting the connection between a person himself (the matrix diagonal) as 0
        ids = np.arange(self.size)
        matrix[ids, ids] = 0

        return matrix

    # todo unify family and workplace creation
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
        self.logger.info(f"Create intra workplace connections")
        matrix = m_type((self.size, self.size), dtype=np.float32)

        # creating all families, and assigning each agent to a family, and counterwise
        agents_without_work = list(range(self.size))
        shuffle(agents_without_work)
        works = []
        num_of_workplaces = self.size // self.consts.average_work_size
        for i in range(num_of_workplaces):  # todo add last work
            if i % (num_of_workplaces // 100) == 0:
                self.logger.info(f"Creating workplace {i}/{num_of_workplaces}")
            new_work = TrackingCircle()
            for _ in range(self.consts.average_work_size):
                chosen_agent_ind = agents_without_work.pop()
                chosen_agent = self.agents[chosen_agent_ind]
                chosen_agent.add_work(new_work)
                new_work.add_agent(chosen_agent)
            works.append(new_work)
        self.works = works

        # adding the remaining people to a work (if size % average_work_size != 0)
        if len(agents_without_work) > 0:
            self.logger.info("adding remaining agents to workplaces")
            new_work = TrackingCircle()
            for agent_index in agents_without_work:
                chosen_agent = self.agents[agent_index]
                chosen_agent.add_work(new_work)
                new_work.add_agent(chosen_agent)
            works.append(new_work)

        # updating the matrix using the works
        for work in works:
            ids = np.array([a.index for a in work.agents])
            xs, ys = np.meshgrid(ids, ids)
            xs = xs.reshape(-1)
            ys = ys.reshape(-1)
            matrix[xs, ys] = self.consts.work_strength
        ids = np.arange(self.size)
        matrix[ids, ids] = 0
        return matrix

    def _create_random_connectivity(self):
        """
        plug here random connection, super spreaders, whatever. We can also adjust the number of daily connections
        b or beta in the literature) by adding this random edges
        :return: lil_matrix n*n
        """
        self.logger.info(f"Create random connections")

        matrix = m_type((self.size, self.size), dtype=np.float32)
        amount_of_connections = self.consts.average_amount_of_strangers
        stranger_ids = np.random.randint(
            0, self.size - 1, self.size * amount_of_connections
        )
        ids = np.arange(self.size).repeat(amount_of_connections)

        matrix[ids, stranger_ids] = self.consts.stranger_strength
        """
        amount_of_connections = social_stats.average_amount_of_strangers
        dense = amount_of_connections / self.size
        matrix = sparse.rand(self.size, self.size, dense)
        matrix.data[:] = social_stats.stranger_strength
        """
        return matrix

    def normalize(self):
        """
        this funciton should normalize the weights within W to represent the infection rate.
        As r0=bd, where b is number of daily infections per person
        """
        self.logger.info(f"normalizing matrix")
        if self.factor is None:
            r0 = self.consts.r0
            # changes r0 to fit the infection ratio (he is calculated despite the low infection ratio)
            r0 = r0 * (1 / self.consts.expected_infection_ratio())
            non_zero_elements = self.matrix.count_nonzero()
            # average number of connections per person per day
            b = non_zero_elements / self.size
            # average probability for infection in each meeting as should be
            d = r0 / (self.consts.average_infecting_days() * b)
            # avarage probability for infection in each meeting in current matrix
            average_edge_weight_in_matrix = self.matrix.sum() / non_zero_elements
            # saves this so that connections will be easily re-established later on
            self.factor = d / average_edge_weight_in_matrix
        self.matrix = (
            self.matrix * self.factor
        )  # now each entry in W is such that bd=R0

        # switching from probability to ln(1-p):
        non_zero_keys = self.matrix.nonzero()
        self.matrix[non_zero_keys] = np.log(1 - self.matrix[non_zero_keys])

    def change_work_policy(self, state):
        self.matrix = self.m_families + state * self.m_work + self.m_random
        self.normalize()
