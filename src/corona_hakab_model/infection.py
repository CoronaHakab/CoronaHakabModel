from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

from common.agent import Agent
from generation import connection_types

if TYPE_CHECKING:
    from manager import SimulationManager


class InfectionInfo:
    def __init__(self, infector_agent, connection_type):
        self.infector_agent = infector_agent
        self.connection_type = connection_type


class InfectionManager:
    """
    Manages the infection stage
    """

    def __init__(self, sim_manager: SimulationManager):
        self.manager = sim_manager

    def infection_step(self) -> Dict[Agent, InfectionInfo]:
        """
        performs an infection step.
        returns a dict of agent_index to InfectionInfo objects.
        """
        # perform infection
        infections: Dict[Agent, InfectionInfo] = self._perform_infection()
        # note - this may overwrite
        infections.update(self._infect_random_connections())

        return infections

    def _perform_infection(self) -> Dict[Agent, InfectionInfo]:
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
        return {self.manager.agents[agent_index]: self._get_infection_info(int(agent_index), v) for agent_index in
                infected_indices}

    def _infect_random_connections(self) -> Dict[Agent, InfectionInfo]:
        connections = self.manager.num_of_random_connections * self.manager.random_connections_factor

        probs_not_infected_from_random_connection = self._get_probs_not_infected_from_random_connection(connections)
        probs_not_infected_from_geo_random_connection = self._get_probs_not_infected_from_geo_random_connection(connections)

        # every agent can have up to one value for each connection type from both vectors.
        # so, if he has a value for a connection type in one vector,
        # the value for the same connection type in the other vector will be 1.
        # adding both vectors and subtracting 1 will preserve the value that was set in the vector.
        probs_not_infected_from_connection = (probs_not_infected_from_random_connection + probs_not_infected_from_geo_random_connection) - 1

        not_infected_probs = np.power(probs_not_infected_from_connection, connections)
        prob_infected_in_any_circle = 1 - not_infected_probs.prod(axis=1)
        infections = self.manager.susceptible_vector & \
                     (np.random.random(len(self.manager.agents)) < prob_infected_in_any_circle)

        infected_indices = np.flatnonzero(infections)
        return {self.manager.agents[agent_index]: self._get_random_infection_info(int(agent_index),
                                                                                  1 - not_infected_probs[agent_index])
                for agent_index in infected_indices}

    def _get_probs_not_infected_from_random_connection(self, connections):
        probs_not_infected_from_random_connection = np.ones_like(connections, dtype=float)

        for connection_type in connection_types.With_Random_Connections:
            for circle in self.manager.social_circles_by_connection_type[connection_type]:
                agents_id = [agent.index for agent in circle.agents]

                if len(agents_id) == 1:
                    # One-Agent circle, you can't randomly meet yourself..
                    continue

                total_infectious_random_connections = np.dot(
                    self.manager.contagiousness_vector[agents_id],
                    connections[agents_id, connection_type],
                )

                prob = total_infectious_random_connections / circle.total_random_connections

                probs_not_infected_from_random_connection[agents_id, connection_type] = \
                    1 - prob * self.manager.random_connections_strength[connection_type]

        return probs_not_infected_from_random_connection

    def _get_probs_not_infected_from_geo_random_connection(self, connections):
        probs_not_infected_from_geo_random_connection = np.ones_like(connections, dtype=float)

        for connection_type in connection_types.With_Geo_Random_Connections:
            for geographic_circle in self.manager.geographic_circles:
                circles = geographic_circle.connection_type_to_social_circles[connection_type]
                agents_id = [agent.index for circle in circles for agent in circle.agents]

                if len(agents_id) == 1:
                    # One-Agent in circles, you can't randomly meet yourself..
                    continue

                total_infectious_random_connections = np.dot(
                    self.manager.contagiousness_vector[agents_id],
                    connections[agents_id, connection_type],
                )

                circles_total_random_connections = np.sum([circle.total_random_connections for circle in circles])
                prob = total_infectious_random_connections / circles_total_random_connections

                probs_not_infected_from_geo_random_connection[agents_id, connection_type] = \
                    1 - prob * self.manager.random_connections_strength[connection_type]

        return probs_not_infected_from_geo_random_connection

    def _get_infection_info(self, agent_index: int, possible_infectors: np.ndarray):
        if not self.manager.consts.backtrack_infection_sources:
            return None

        infection_cases = []
        infection_probabilities = []
        non_zero_column = self.manager.matrix.non_zero_column(agent_index)
        for connection_type in connection_types.ConnectionTypes:
            possible_cases = [other_index for other_index in non_zero_column[connection_type] if
                              possible_infectors[other_index]]

            infection_cases += [InfectionInfo(self.manager.agents[infector_index], connection_type) for infector_index
                                in possible_cases]
            infection_probabilities += [self.manager.matrix.get(connection_type, agent_index, infector_index)
                                        for infector_index in possible_cases]

        norm_factor = sum(infection_probabilities)
        infection_probabilities = [ip / norm_factor for ip in infection_probabilities]
        return np.random.choice(infection_cases, p=infection_probabilities)

    def _get_random_infection_info(self, agent_id: int, infection_probs: np.ndarray):
        if not self.manager.consts.backtrack_infection_sources:
            return None
        
        # determine infection method
        connection_type = np.random.choice(len(infection_probs), p=infection_probs / np.sum(infection_probs))

        # determine infector
        agents_id = []

        if connection_type in connection_types.With_Random_Connections:
            infectious_circle = next(circle for circle in self.manager.social_circles_by_agent_index[agent_id]
                                     if circle.connection_type == connection_type)
            agents_id = [agent.index for agent in infectious_circle.agents]

        if connection_type in connection_types.With_Geo_Random_Connections:
            # TODO: if we see that this part of code is slow -
            #  add dict of agents and the geo circle that they work in(should be created in the generation process)
            agent_connection_type_circle = next(circle for circle in self.manager.social_circles_by_agent_index[agent_id]
                                                if circle.connection_type == connection_type)
            agent_geo_circle = next(geo_circle for geo_circle in self.manager.geographic_circles
                                    if agent_connection_type_circle in geo_circle.connection_type_to_social_circles[connection_type])
            infectious_circles = [circle for circle in agent_geo_circle.connection_type_to_social_circles[connection_type]]
            agents_id = [agent.index for circle in infectious_circles for agent in circle.agents]

        connections = self.manager.num_of_random_connections * self.manager.random_connections_factor
        infectious_agents = self.manager.contagiousness_vector[agents_id] * connections[agents_id, connection_type]
        infector_id = np.random.choice(agents_id, p=infectious_agents / np.sum(infectious_agents))
        return InfectionInfo(self.manager.agents[infector_id], connection_type)
