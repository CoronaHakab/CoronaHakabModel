from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

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
        v = np.random.random(self.manager.agents_df.n_agents()) < self.manager.agents_df.contagious_vec()

        # u = mat dot_product v (log of the probability that an agent will get infected)
        u = self.manager.matrix.prob_any(v)
        # calculate the infections boolean vector

        infections = self.manager.agents_df.susceptible_vec() & (np.random.random(u.shape) < u)
        infected_indices = np.flatnonzero(infections)

        # new_infected: dict -
        # key = medical state (currently only susceptible state which an agent can be infected)
        # value = list of agents
        return infected_indices
