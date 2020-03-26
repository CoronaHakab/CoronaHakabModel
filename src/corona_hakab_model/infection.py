from collections import defaultdict

import manager
import numpy as np


class InfectionManager:
    """
    Manages the infection stage
    """

    def __init__(self, sim_manager: "manager.SimulationManager"):
        self.agents_to_home_isolation = []
        self.agents_to_full_isolation = []
        self.manager = sim_manager

    def infection_step(self):
        # perform infection
        self.agents_to_home_isolation.clear()
        self.agents_to_full_isolation.clear()
        return self._perform_infection()

    def _perform_infection(self):
        """
        perform the infection stage by multiply matrix with infected vector and try to infect agents.

        v = [i for i in self.agents.is_contagious]
        perform w*v
        for each person in v:
            if rand() < v[i]
                agents[i].infect
        """

        # v = [True if an agent can infect other agents in this time step]
        v = np.random.random(len(self.manager.agents)) < self.manager.contagiousness_vector

        # u = mat dot_product v (log of the probability that an agent will get infected)
        u = self.manager.matrix.matrix.dot(v)
        # calculate the infections boolean vector
        infections = self.manager.susceptible_vector & (np.random.random(u.shape) < (1 - np.exp(u)))
        infected_indices = np.flatnonzero(infections)

        # caught_rolls: boolean vector, True if an agent is known to be infected
        # thus the authorities could act upon that
        caught_rolls = np.random.random(len(infected_indices)) < self.manager.consts.caught_sicks_ratio

        # new_infected: dict -
        # key = medical state (currently only susceptible state which an agent can be infected)
        # value = list of agents
        new_infected = defaultdict(list)
        for index, caught in zip(infected_indices, caught_rolls):
            agent = self.manager.agents[index]
            new_infected[agent.medical_state].append(agent)
            if caught:
                # what to do with an infected agent that got caught
                if self.manager.consts.home_isolation_sicks:
                    self.agents_to_home_isolation.append(agent)
                elif self.manager.consts.full_isolation_sicks:
                    self.agents_to_full_isolation.append(agent)

        return new_infected
