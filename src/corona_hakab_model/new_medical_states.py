from __future__ import annotations
from agent import Agent
from abc import ABC
from consts import Consts

class MedicalState(ABC):
    """
    Basic state in a medical state machine -
    Every Agent will have a .current_state member of this type.
    Every subclass must produce the following:

    duration:       Amount of steps left in this state.
                    Depends on the Agent (age, TODO medical history, medical care, other?)
    contagiousness: How contagious the Agent is (in the contagiousness vector)
                    Depends on the Agent (age, TODO other?)
    next_state():   The next state when duration is over.
                    Depends Agent (age, TODO medical history, medical care, other?)
    """

    agents_count = 0
    agents_set = set()

    def __init__(self, agent: Agent):
        self.agent = agent
        self.duration = self.get_duration()
        self.contagiousness = self.get_contagiousness()
        self.simulation_manager = agent.simulation_manager

    def get_duration(self):
        """
        Based on the state's parameters and self.agent.traits
        """
        pass

    def get_contagiousness(self):
        """
        Based on the state's parameters and self.agent.traits
        """
        pass

    def next_state(self):
        """
        Based on the state's parameters and self.agent.traits
        """
        pass


class SusceptibleState(MedicalState):
    """
    Susceptible - agent has not been infected yet
    """

    def get_duration(self):
        """
        Agents are susceptible until someone actively infects them.
        """
        return float("inf")

    def get_contagiousness(self):
        """
        Cannot infect anyone
        """
        return 0

    def get_next_state(self):
        transfers = self.simulation_manager.consts.susceptible_state_transfers
        


