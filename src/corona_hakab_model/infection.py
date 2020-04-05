from collections import defaultdict
from typing import Dict, List

import manager
import numpy as np
from medical_state import MedicalState


class InfectionManager:
    """
    Manages the infection stage
    """

    def __init__(self, sim_manager: "manager.SimulationManager"):
        self.manager = sim_manager

    def infection_step(self) -> Dict[MedicalState, List]:
        # perform infection
        return self._perform_infection()

    def _perform_infection(self) -> Dict[MedicalState, List]:
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
        u = self.manager.matrix.prob_any(v)
        # calculate the infections boolean vector

        infections = self.manager.susceptible_vector & (np.random.random(u.shape) < u)
        infected_indices = np.flatnonzero(infections)

        # new_infected: dict -
        # key = medical state (currently only susceptible state which an agent can be infected)
        # value = list of agents
        new_infected = defaultdict(list)
        for index in infected_indices:
            agent = self.manager.agents[index]
            new_infected[agent.medical_state].append(agent)

        # detected_daily keeps the amount of agents that got detected in the current step
        # each silent agent has detection_rate chance of being detected in each step.
        self.manager.detected_daily = round(self.manager.consts.detection_rate * self.manager.in_silent_state)
        # the detected agents from this step are deducted from the total silent agents
        self.manager.in_silent_state -= self.manager.detected_daily

        return new_infected
