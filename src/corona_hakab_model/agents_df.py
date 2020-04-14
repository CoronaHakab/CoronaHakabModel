from typing import List, Set

import numpy as np
import pandas as pd

from agent import Agent
from detection_model.detection_testing_types import DetectionSettings
from detection_model.healthcare import DetectionResult
from medical_state import MedicalState
from medical_state_machine import MedicalStateMachine


class AgentsDf:
    def __init__(self, agents: List[Agent], state_machine: MedicalStateMachine):
        self._df = pd.DataFrame(
            columns=['age', 'medical_state', 'test_willingness', 'susceptible', 'contagious', 'test_result',
                     'ever_tested_positive', 'date_of_last_test'], index=[a.index for a in agents])
        # initial data
        self.state_machine = state_machine
        self._df.age = [a.age for a in agents]
        self._df.medical_state = state_machine.initial
        self._df.test_willingness = 0.0
        self._df.susceptible = False
        self._df.ever_tested_positive = False
        self._df.contagious = 0
        self._df.test_result = DetectionResult.NOT_TAKEN
        self._df.date_of_last_test = 0
        print(self._df.head())

    def n_agents(self) -> int:
        return self._df.shape[0]

    def agents_ind(self) -> List[int]:
        return self._df.index.values.tolist()

    def change_agents_state(self, index_list: List[int], state: MedicalState) -> None:
        col_names = ['medical_state', 'test_willingness', 'contagious', 'susceptible']
        col_vals = [state, state.test_willingness, state.contagiousness, state.susceptible]
        self._df.loc[index_list, col_names] = col_vals

    def at(self, index_list):
        return self._df.loc[index_list].copy()

    def test_candidates(self, test_location: DetectionSettings, curr_date: int) -> Set[int]:
        def want_to_be_tested():
            random_vec = np.random.random(self.n_agents())
            return self._df.test_willingness > random_vec

        def alive():
            return self._df.medical_state != self.state_machine.states_by_name['Deceased']

        def can_be_tested_again():
            neg_gap = test_location.testing_gap_after_negative_test
            pos_gap = test_location.testing_gap_after_positive_test

            # never tested
            return ((self._df.test_result == DetectionResult.NOT_TAKEN) |
                    # tested negative but gap passed
                    ((self._df.test_result == DetectionResult.NEGATIVE) &
                     (self._df.date_of_last_test > curr_date - neg_gap)) |
                    # tested positive but gap passed
                    ((self._df.test_result == DetectionResult.POSITIVE) &
                     (self._df.date_of_last_test > curr_date - pos_gap))
                    )

        return set(self._df[want_to_be_tested()][alive()][can_be_tested_again()].index.values.tolist())

    def contagious_vec(self):
        return self._df.contagious.tolist()

    def susceptible_vec(self):
        return self._df.susceptible.tolist()

    def set_test_start(self, agent_index, current_step):
        self._df.loc[agent_index].date_of_last_test = current_step

    def set_test_result(self, agent_ind, test_result):
        self._df.loc[agent_ind, 'test_result'] = test_result
        if test_result == DetectionResult.POSITIVE:
            self._df.loc[agent_ind, 'ever_tested_positive'] = True

    def ever_tested_positive(self, ind):
        return self.at(ind).ever_tested_positive
