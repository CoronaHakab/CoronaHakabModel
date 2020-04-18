from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from agent import Agent
from generation.connection_types import ConnectionTypes

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
        
        # get infection methods
        infection_methods = []
        nonzero_columns = self.manager.matrix.non_zero_columns()
        for agent_index in infected_indices:
            infection_methods.append(self._get_infection_method(agent_index, nonzero_columns))
            
        # new_infected: dict -
        # key = medical state (currently only susceptible state which an agent can be infected)
        # value = list of agents
        new_infected = list()
        for index in infected_indices:
            agent = self.manager.agents[index]
            new_infected.append(agent)

        return new_infected, infection_methods

    def _get_infection_method(self, agent_index, non_zero_columns):
        probability_by_conn_type = []
        for connection_type in ConnectionTypes:
            possible_infectors = non_zero_columns[connection_type][agent_index]
            conn_infection_probability = sum([self.manager.matrix.get(
                int(connection_type), 
                int(agent_index), 
                other_index) 
                for other_index in possible_infectors])
            probability_by_conn_type.append(conn_infection_probability)
        normalize_factor = 1/sum(probability_by_conn_type)
        probs_by_conn_type = [prob * normalize_factor for prob in probability_by_conn_type]
        return np.random.choice(ConnectionTypes, p=probs_by_conn_type)    