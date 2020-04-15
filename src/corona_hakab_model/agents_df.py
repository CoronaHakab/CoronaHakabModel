from typing import List, Set, Iterable, Union, Sequence

import numpy as np
import pandas as pd

from agent import Agent
from detection_model.detection_testing_types import DetectionSettings
from detection_model.healthcare import DetectionResult
from medical_state import MedicalState
from medical_state_machine import MedicalStateMachine


class AgentsDf:
    def __init__(self, agents: List[Agent], state_machine: MedicalStateMachine):
        self._df: pd.DataFrame = pd.DataFrame(
            columns=['age', 'medical_state', 'test_willingness', 'susceptible', 'contagious', 'detectable',
                     'test_result', 'ever_tested_positive', 'date_of_last_test'], index=[a.index for a in agents])
        # initial data
        self.state_machine = state_machine
        self._df.age = [a.age for a in agents]
        initial: MedicalState = state_machine.initial
        self._df.medical_state = initial
        self._df.test_willingness = initial.test_willingness
        self._df.susceptible = initial.susceptible
        self._df.contagious = initial.contagiousness
        self._df.detectable = initial.detectable
        self._df.ever_tested_positive = False
        self._df.test_result = DetectionResult.NOT_TAKEN
        self._df.date_of_last_test = 0
        self._cols_to_idx = {col: i for i, col in enumerate(self._df.columns)}

    def n_agents(self) -> int:
        return self._df.shape[0]

    def agents_indices(self) -> Iterable[int]:
        return self._df.index.values

    def change_agents_state(self, index_list: List[int], states: Union[MedicalState, Iterable[MedicalState]]) -> None:
        col_names = ['medical_state', 'test_willingness', 'contagious', 'susceptible', 'detectable']

        if not isinstance(states, Iterable):
            # all agents will have the same state
            states = [states]

        values = [[state, state.test_willingness, state.contagiousness, state.susceptible, state.detectable]
                  for state in states]

        self._df.loc[index_list, col_names] = values

    def at(self, index_list):
        return self._df.loc[index_list]

    def test_candidates(self, test_location: DetectionSettings, curr_date: int) -> Set[int]:
        def want_to_be_tested():
            random_vec = np.random.random(self.n_agents())
            return self._df.test_willingness > random_vec

        def alive():
            return self._df.medical_state != self.state_machine.states_by_name['Deceased']

        def can_be_tested_again():
            neg_gap = test_location.testing_gap_after_negative_test
            pos_gap = test_location.testing_gap_after_positive_test

            return (
                # never tested
                    (self._df.test_result == DetectionResult.NOT_TAKEN) |
                    # tested negative but gap passed
                    ((self._df.test_result == DetectionResult.NEGATIVE) &
                     (self._df.date_of_last_test > curr_date - neg_gap)) |
                    # tested positive but gap passed
                    ((self._df.test_result == DetectionResult.POSITIVE) &
                     (self._df.date_of_last_test > curr_date - pos_gap))
            )

        return set(self._df.index[want_to_be_tested() & alive() & can_be_tested_again()])

    def contagious_vec(self):
        return self._df.contagious.values

    def susceptible_vec(self):
        return self._df.susceptible.values

    def set_tests_start(self, agent_index: Union[int, Iterable[int]], current_step: int):
        if isinstance(agent_index, (int, np.integer)):
            self._df.at[agent_index, 'date_of_last_test'] = current_step
        else:
            self._df.loc[agent_index, 'date_of_last_test'] = current_step

    def set_tests_results(self, agent_index: Union[int, Sequence[int]],
                          test_result: Union[DetectionResult, Iterable[DetectionResult]]):
        if isinstance(agent_index, (int, np.integer)):
            self._df.at[agent_index, 'test_result'] = test_result
            if test_result == DetectionResult.POSITIVE:
                self._df.at[agent_index, 'ever_tested_positive'] = True
        else:
            self._df.loc[agent_index, 'test_result'] = test_result
            self._df.loc[agent_index[test_result == DetectionResult.POSITIVE], 'ever_tested_positive'] = True

    def ever_tested_positive(self, agent_index: Union[int, Iterable[int]]) -> Union[bool, Iterable[bool]]:
        if isinstance(agent_index, (int, np.integer)):
            return self._df.at[agent_index, 'ever_tested_positive']
        else:
            return self._df.loc[agent_index, 'ever_tested_positive'].values
