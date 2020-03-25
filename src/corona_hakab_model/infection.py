from collections import defaultdict

import manager
import numpy as np


class InfectionManager:
    """
    Manages the infection stage
    """

    def __init__(self, sim_manager: "manager.SimulationManager"):
        self.agents_to_home_quarantine = []
        self.agents_to_full_quarantine = []
        self.manager = sim_manager

    def infection_step(self):
        # perform infection
        self.agents_to_home_quarantine.clear()
        self.agents_to_full_quarantine.clear()
        return self._perform_infection()

    def _perform_infection(self):
        """
        perform the infection stage by multiply matrix with infected vector and try to infect agents.

        v = [i for i in self.agents.is_infectious]
        perform w*v
        for each person in v:
            if rand() < v[i]
                agents[i].infect
        """

        v = (
            np.random.random(len(self.manager.agents))
            < self.manager.infectiousness_vector
        )

        u = self.manager.matrix.matrix.dot(v)
        infections = self.manager.infectable_vector & (
            np.random.random(u.shape) < (1 - np.exp(u))
        )
        infected_indices = np.flatnonzero(infections)

        caught_rolls = (
            np.random.random(len(infected_indices))
            < self.manager.consts.caught_sicks_ratio
        )
        new_infected = defaultdict(list)
        for index, caught in zip(infected_indices, caught_rolls):
            agent = self.manager.agents[index]
            new_infected[agent.medical_state].append(agent)
            if caught:
                if self.manager.consts.home_quarantine_sicks:
                    self.agents_to_home_quarantine.append(agent)
                elif self.manager.consts.full_quarantine_sicks:
                    self.agents_to_full_quarantine.append(agent)

        return new_infected
