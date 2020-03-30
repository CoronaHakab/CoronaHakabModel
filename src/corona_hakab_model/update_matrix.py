import numpy as np
from agent import Agent


class UpdateMatrixManager:
    """
    Manages the "Update Matrix" stage of the simulation.
    """

    def __init__(self, affinity_matrix_ref):
        self.affinity_matrix = affinity_matrix_ref

    def update_matrix_step(self, agents_to_home_isolation=(), agents_to_full_isolation=()):
        """
        Update the matrix step
        """
        # for now, we will not update the matrix at all
        # self.apply_self_isolation(matrix, family_matrix, sick_agents_vector, agents_list)
        for agent in agents_to_home_isolation:
            self.home_isolation_agent(agent)
        for agent in agents_to_full_isolation:
            self.full_isolation_agent(agent)
        return

    def home_isolation_agent(self, agent: Agent):
        """
        gets and agent and puts him in home isolation.
        updates the matrix accordingly
        """
        if agent.is_home_isolated:
            return
        families = self.affinity_matrix.m_families
        # changing your col (now you won't infect any one outside of your home)
        indices = (
            np.full(self.affinity_matrix.size, agent.ID, dtype=int),
            np.arange(self.affinity_matrix.size),
        )
        temp = 1 - (families[indices] * self.affinity_matrix.factor)
        self.affinity_matrix.matrix[indices] = np.log(temp)

        # changing your row (now you won't be infected by people outside of your home)
        indices = (indices[1], indices[0])
        temp = 1 - (families[indices] * self.affinity_matrix.factor)
        self.affinity_matrix.matrix[indices] = np.log(temp)

        agent.is_home_isolated = True

    def full_isolation_agent(self, agent: Agent):
        """
        gets and agent and puts him in home isolation.
        updates the matrix accordingly
        """
        if agent.is_full_isolated:
            return
        # changing your col (now you won't infect any one)
        indices = (
            np.full(self.affinity_matrix.size, agent.ID, dtype=int),
            np.arange(self.affinity_matrix.size),
        )
        self.affinity_matrix.matrix[indices] = 0

        # changing your row (now you won't be infected by people)
        indices = (indices[1], indices[0])
        self.affinity_matrix.matrix[indices] = 0

        agent.is_full_isolated = True

    def remove_agent_from_isolation(self, agent: Agent):
        """
        removes an agent from home isolation
        updates the matrix accordingly
        """
        if not agent.is_home_isolated:
            return
        # changing your col (now you will infect people outside of your home)
        families = self.affinity_matrix.m_families
        works = self.affinity_matrix.m_work
        random = self.affinity_matrix.m_random
        indices = (
            np.full(self.affinity_matrix.size, agent.ID, dtype=int),
            np.arange(self.affinity_matrix.size),
        )
        temp = 1 - ((families[indices] + works[indices] + random[indices]) * self.affinity_matrix.factor)
        self.affinity_matrix.matrix[indices] = np.log(temp)

        # changing your row (now you will be infected by people outside your home)
        indices = (indices[1], indices[0])
        temp = 1 - ((families[indices] + works[indices] + random[indices]) * self.affinity_matrix.factor)
        self.affinity_matrix.matrix[indices] = np.log(temp)

        agent.is_home_isolated = False
        agent.is_full_isolated = False
