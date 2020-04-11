from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from agent import Agent

from agent import Agent

if TYPE_CHECKING:
    from manager import SimulationManager


class InfectionManager:
    """
    Manages the infection stage
    """

    def __init__(self, sim_manager: SimulationManager):
        self.manager = sim_manager

    def infection_step(self) -> List[Agent]:
        # perform infection
        return self._perform_infection()

    def _perform_infection(self) -> List[Agent]:
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
        new_infected = list()
        for index in infected_indices:
            agent = self.manager.agents[index]
            new_infected.append(agent)

        return new_infected
