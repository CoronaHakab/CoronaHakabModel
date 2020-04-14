from functools import lru_cache
from itertools import count
from typing import Dict, List
import numpy as np

from medical_state import MedicalState
from state_machine import StateMachine
from util import upper_bound


class MedicalStateMachine(StateMachine[MedicalState]):
    def __init__(self, initial_state: MedicalState, default_state_upon_infection: MedicalState, sick_states: List[str],
                 was_ever_sick_states: List[str]):
        super().__init__(initial_state)
        self.default_state_upon_infection = default_state_upon_infection
        self.add_state(default_state_upon_infection)
        self.sick_states = sick_states
        self.was_ever_sick_states = was_ever_sick_states

    def get_state_upon_infection(self, agent) -> MedicalState:
        if agent:  # placeholder
            pass
        return self.default_state_upon_infection
        # todo add virtual link between initial and infected state for graphs

    @lru_cache(None)
    def average_time_in_each_state(self) -> Dict[MedicalState, int]:
        """
        calculate the average time an infected agent spends in any of the states.
        uses markov chain to do the calculations
        note that it doesnt work well for terminal states
        :return: dict of states: int, representing the average time an agent would be in a given state
        """
        TOL = 1e-6
        M, terminal_states, transfer_states, entry_columns = self.markovian
        z = len(M)

        p = entry_columns[self.default_state_upon_infection]
        terminal_mask = np.zeros(z, bool)
        terminal_mask[list(terminal_states.values())] = True

        states_duration: Dict[MedicalState, int] = Dict.fromkeys(self.states, 0)
        states_duration[self.default_state_upon_infection] = 1

        index_to_state: Dict[int, MedicalState] = {}
        for state, index in terminal_states.items():
            index_to_state[index] = state
        for state, dict in transfer_states.items():
            first_index = dict[0]
            last_index = dict[max(dict.keys())] + upper_bound(state.durations[-1])
            for index in range(first_index, last_index):
                index_to_state[index] = state

        prev_v = 0.0
        for time in count(1):
            p = M @ p
            v = np.sum(p, where=terminal_mask)
            d = v - prev_v
            prev_v = v

            for i, prob in enumerate(p):
                states_duration[index_to_state[i]] += prob

            # run at least as many times as the node number to ensure we reached all terminal nodes
            if time > z and d < TOL:
                break
        return states_duration
