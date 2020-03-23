from affinity_matrix import AffinityMatrix
from agent import Agent
import numpy as np


class UpdateMatrixManager:
    """
    Manages the "Update Matrix" stage of the simulation.
    """

    def __init__(self, affinity_matrix_ref):
        self.affinity_matrix = affinity_matrix_ref

    def update_matrix_step(self, agents_to_home_quarantine=(), agents_to_full_quarantine=()):
        """
        Update the matrix step
        """
        # for now, we will not update the matrix at all
        # self.apply_self_quarantine(matrix, family_matrix, sick_agents_vector, agents_list)
        for agent in agents_to_home_quarantine:
            self.home_quarantine_agent(agent)
        for agent in agents_to_full_quarantine:
            self.full_quarantine_agent(agent)
        return

    def home_quarantine_agent(self, agent: Agent):
        """
        gets and agent and puts him in home quarenite.
        updates the matrix accordingly
        """
        if agent.is_home_quarantined:
            return
        families = self.affinity_matrix.m_families
        # changing your col (now you won't infect any one outside of your home)
        indices = np.full(self.affinity_matrix.size, agent.ID, dtype=int), np.arange(self.affinity_matrix.size)
        temp = (1 - (families[indices] * self.affinity_matrix.factor))
        self.affinity_matrix.matrix[indices] = np.log(temp)

        # changing your row (now you won't be infected by people outside of your home)
        indices = (indices[1], indices[0])
        temp = (1 - (families[indices] * self.affinity_matrix.factor))
        self.affinity_matrix.matrix[indices] = np.log(temp)

        agent.is_home_quarantined = True

    def full_quarantine_agent(self, agent: Agent):
        """
        gets and agent and puts him in home quarenite.
        updates the matrix accordingly
        """
        if agent.is_full_quarantined:
            return
        # changing your col (now you won't infect any one)
        indices = np.full(self.affinity_matrix.size, agent.ID, dtype=int), np.arange(self.affinity_matrix.size)
        self.affinity_matrix.matrix[indices] = 0

        # changing your row (now you won't be infected by people)
        indices = (indices[1], indices[0])
        self.affinity_matrix.matrix[indices] = 0

        agent.is_full_quarantined = True

    def remove_agent_from_quarantine(self, agent: Agent):
        """
        removes an agent from home quarantine
        updates the matrix accordingly
        """
        if not agent.is_home_quarantined:
            return
        # changing your col (now you will infect people outside of your home)
        families = self.affinity_matrix.m_families
        works = self.affinity_matrix.m_work
        random = self.affinity_matrix.m_random
        indices = np.full(self.affinity_matrix.size, agent.ID, dtype=int), np.arange(self.affinity_matrix.size)
        temp = (1 - ((families[indices] + works[indices] + random[indices]) * self.affinity_matrix.factor))
        self.affinity_matrix.matrix[indices] = np.log(temp)

        # changing your row (now you will be infected by people outside your home)
        indices = (indices[1], indices[0])
        temp = (1 - ((families[indices] + works[indices] + random[indices]) * self.affinity_matrix.factor))
        self.affinity_matrix.matrix[indices] = np.log(temp)

        agent.is_home_quarantined = False
        agent.is_full_quarantined = False

    def apply_self_quarantine(self, matrix, family_matrix, sick_agents_vector, agents_list):  # not in use
        """
        Modifies the matrix so self quarantines are in place
        """
        sick_agents_ids = sick_agents_vector.get_sick_ids()

        # run on all self quarantined agents
        for agent_id in sick_agents_ids:
            if agents_list[agent_id].is_self_quarantined():
                # remove all existing relations
                matrix.zero_column(agent_id)
                # insert only family relations (as he is at home)
                matrix.add_to_column(agent_id, family_matrix.get_column(agent_id))
