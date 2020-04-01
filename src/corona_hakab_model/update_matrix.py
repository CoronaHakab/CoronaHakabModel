from affinity_matrix import AffinityMatrix
from agent import Agent


class UpdateMatrixManager:
    """
    Manages the "Update Matrix" stage of the simulation.
    """

    def __init__(self, affinity_matrix_ref: AffinityMatrix):
        self.affinity_matrix = affinity_matrix_ref

    def update_matrix_step(self, agents_to_home_isolation=(), agents_to_full_isolation=()):
        """
        Update the matrix step
        """
        # for now, we will not update the matrix at all
        # self.apply_self_isolation(matrix, family_matrix, sick_agents_vector, agents_list)
        if not agents_to_home_isolation and not agents_to_full_isolation:
            return
        with self.affinity_matrix.inner.lock_rebuild():
            for agent in agents_to_home_isolation:
                self.home_isolation_agent(agent)
            for agent in agents_to_full_isolation:
                self.full_isolation_agent(agent)

    def home_isolation_agent(self, agent: Agent):
        """
        gets and agent and puts him in home isolation.
        updates the matrix accordingly
        """
        if agent.is_home_isolated:
            return
        fam_index = self.affinity_matrix.circular_matrix_types["home"]
        for i in range(self.affinity_matrix.depth):
            if i == fam_index:
                continue
            self.affinity_matrix.inner.mul_sub_col(i, agent.index, 0)
            self.affinity_matrix.inner.mul_sub_row(i, agent.index, 0)

        agent.is_home_isolated = True

    def full_isolation_agent(self, agent: Agent):
        """
        gets and agent and puts him in home isolation.
        updates the matrix accordingly
        """
        if agent.is_full_isolated:
            return
        # changing your col (now you won't infect any one)
        for i in range(self.affinity_matrix.depth):
            self.affinity_matrix.inner.mul_sub_col(i, agent.index, 0)
            self.affinity_matrix.inner.mul_sub_row(i, agent.index, 0)

        agent.is_full_isolated = True

    def remove_agent_from_isolation(self, agent: Agent):
        """
        removes an agent from home isolation
        updates the matrix accordingly
        """
        if not agent.is_home_isolated:
            return

        for i in range(self.affinity_matrix.depth):
            self.affinity_matrix.inner.reset_mul_col(i, agent.index)
            self.affinity_matrix.inner.reset_mul_row(i, agent.index)

        agent.is_home_isolated = False
        agent.is_full_isolated = False
