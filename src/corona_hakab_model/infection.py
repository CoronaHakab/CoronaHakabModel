from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

from agent import Agent
from generation import connection_types
import agent

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
        infections = self._perform_infection()
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
        nonzero_columns = self.manager.matrix.non_zero_columns()
        return {self.manager.agents[agent_index] : self._get_infection_info(agent_index, nonzero_columns, v) for agent_index in infected_indices}
        
    def _infect_random_connections(self) -> Dict[Agent, InfectionInfo]:
        connections = self.manager.num_of_random_connections * self.manager.random_connections_factor
        probs_not_infected_from_connection = np.ones_like(connections, dtype=float)
        for connection_type in connection_types.With_Random_Connections:
            for circle in self.manager.social_circles_by_connection_type[connection_type]:
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
                    1 - prob * self.manager.random_connections_strength[connection_type]

        not_infected_probs = np.power(probs_not_infected_from_connection, connections)
        prob_infected_in_any_circle = 1 - not_infected_probs.prod(axis=1)
        infections = self.manager.susceptible_vector & \
                     (np.random.random(len(self.manager.agents)) < prob_infected_in_any_circle)
                     
        infected_indices = np.flatnonzero(infections)
        return {self.manager.agents[agent_index] : self._get_random_infection_info(agent_index, 1 - not_infected_probs[agent_index])
                for agent_index in infected_indices}
    
    def _get_infection_info(self, agent_index, non_zero_columns, possible_infectors):
        infection_cases = []
        infection_probabilities = []
        for connection_type in connection_types.ConnectionTypes:
            possible_cases = [other_index for other_index in non_zero_columns[connection_type][agent_index] if possible_infectors[other_index]]
            for infector_index in possible_cases:
                infection_cases.append(InfectionInfo(
                    self.manager.agents[infector_index],
                    connection_type))
                infection_probabilities.append(self.manager.matrix.get(
                    int(connection_type),
                    int(agent_index),
                    infector_index))
        infection_probabilities = [prob/sum(infection_probabilities) for prob in infection_probabilities]
        return infection_cases[np.random.choice(len(infection_cases), p=infection_probabilities)]  
    
    def _get_random_infection_info(self, agent_id, infection_probs):
        # determine infection method
        connection_type = np.random.choice(len(infection_probs), p=infection_probs/sum(infection_probs))
        
        # determine infector
        infectious_circle = [c for c in self.manager.social_circles_by_agent_index[agent_id] if c.connection_type == connection_type][0]
        agents_id = [a.index for a in infectious_circle.agents]
        connections = self.manager.num_of_random_connections * self.manager.random_connections_factor
        infectious_agents = self.manager.contagiousness_vector[(agents_id,)] * connections[(agents_id, [connection_type] * len(agents_id))]
        infector_id = agents_id[np.random.choice(len(infectious_agents), p=infectious_agents/sum(infectious_agents))]
        return InfectionInfo(self.manager.agents[infector_id], connection_type)