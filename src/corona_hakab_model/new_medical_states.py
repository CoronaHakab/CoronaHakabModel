from __future__ import annotations

import numpy as np
from agent import Agent
from abc import ABC


__doc__ = """
A basic weighted options choice data-set is serialized as follows:

<STATE>_possible_<OUTCOMES> = {[ {"age_range": {"min": <MINIMUM_AGE>, "max": <MAXIMUM_AGE>},
                                  "probabilities": ["<OUTCOME_A>": <PROBABILITY_A>,
                                                    "<OUTCOME_B>": <PROBABILITY_B>,
                                                    ...]},
                                 {...},
                               ]}
                                                   

"""


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

    __slots__ = ("agent",
                 "duration",
                 "contagiousness",
                 "simulation",
                 "entered_state_step",
                 )
    agents_count = 0
    agents_set = set()

    def __init__(self, agent: Agent):
        self.agent = agent
        self.next_state = self.next_state()
        self.duration = self.get_duration()
        self.contagiousness = self.get_contagiousness()
        self.simulation_manager = agent.simulation_manager
        self.entered_state_step = self.simulation_manager.current_step

    @classmethod
    def set_durations(cls, possible_durations):
        cls.possible_durations = possible_durations

    @classmethod
    def set_transfers(cls, possible_transfers):
        cls.possible_transfers = possible_transfers

    def get_duration(self):
        """
        Based on the state's parameters and self.agent.traits
        """
        for age_range, probabilities in self.possible_durations:
            if age_range[0] <= self.agent.traits.age < age_range[1]:
                break
        else:
            raise ValueError("Age doesn't fit any age-range of possible_durations")

        # TODO add more complicated logic

        np.random.choice(probabilities[0], p=probabilities[1])

    def get_contagiousness(self):
        """
        Based on the state's parameters and self.agent.traits
        """
        pass

    def next_state(self):
        """
        Based on the state's parameters and self.agent.traits
        """
        for age_range, probabilities in self.possible_durations:
            if age_range[0] <= self.agent.traits.age < age_range[1]:
                break
        else:
            raise ValueError("Age doesn't fit any age-range of possible_transfers")

        # TODO add more complicated logic

        np.random.choice(probabilities[0], p=probabilities[1])


class SusceptibleState(MedicalState):
    """
    Susceptible - agent has not been infected yet
    """



