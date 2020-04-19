from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from agent import Agent

if TYPE_CHECKING:
    from medical_state import MedicalState
    from manager import SimulationManager


class InfectionManager:
    """
    Manages the infection stage
    """

    def __init__(self, sim_manager: SimulationManager):
        self.manager = sim_manager

    def infection_step(self) -> List[Agent]:
        # perform infection
        regular_infections = self._perform_infection()
        random_infections = self._infect_random_connections()

        all_infections = regular_infections | random_infections

        infected_agents = [self.manager.agents[i] for i in np.flatnonzero(all_infections)]
        return infected_agents

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
        u = self.manager.matrix.prob_any(v)
        # calculate the infections boolean vector

        infections = self.manager.susceptible_vector & (np.random.random(u.shape) < u)
        return infections

    def _infect_random_connections(self):
        connections = self.manager.num_of_random_connections * self.manager.random_connections_factor
        probs_not_infected_from_connection = np.ones_like(connections, dtype=float)
        for connection_type, all_circs in self.manager.social_circles_by_connection_type.items():
            for circle in all_circs:
                agents_id = [a.index for a in circle.agents]

                if len(agents_id) == 1:
                    # One-Agent circle, you can't randomly meet yourself..
                    continue

                total_infectious_random_connections = np.dot(
                    self.manager.contagiousness_vector[(agents_id,)],
                    connections[(agents_id, [connection_type] * len(agents_id))],
                )

                prob = total_infectious_random_connections / circle.total_random_connections

                probs_not_infected_from_connection[(agents_id, [connection_type] * len(agents_id))] = \
                    1 - prob * circle.random_connections_strength_factor * self.manager.update_matrix_manager.normalize_factor

        not_infected_probs = np.power(probs_not_infected_from_connection, connections)
        prob_infected_in_any_circle = 1 - not_infected_probs.prod(axis=1)
        infections = self.manager.susceptible_vector & \
                     (np.random.random(len(self.manager.agents)) < prob_infected_in_any_circle)
        return infections
