from typing import List, Iterable, Union, Sequence

import numpy as np

from agent import Agent
from detection_model.detection_testing_types import DetectionSettings
from detection_model.healthcare import DetectionResult
from medical_state import MedicalState
from medical_state_machine import MedicalStateMachine


class AgentsDf:
    def __init__(self, agents: List[Agent] = None, state_machine: MedicalStateMachine = None):
        # Enable initiation with empty c'tor call
        if agents is None or state_machine is None:
            self.state_machine, self.age, self.medical_state, self._ever_tested_positive, self._test_result, \
                self._date_of_last_test, self.num_agents, self.index = [None] * 8
            return

        # initial data
        self.state_machine = state_machine
        initial: MedicalState = state_machine.initial

        # no pandas fields
        num_agents = len(agents)
        self.age = np.array([a.age for a in agents])
        self.medical_state: np.ndarray = np.full(num_agents, initial)
        self._ever_tested_positive = np.full(num_agents, False)
        self._test_result = np.full(num_agents, DetectionResult.NOT_TAKEN)
        self._date_of_last_test = np.full(num_agents, 0)
        self.num_agents = num_agents
        self.index = np.array([a.index for a in agents])

    def n_agents(self) -> int:
        return self.num_agents

    def agents_indices(self) -> Iterable[int]:
        return self.index

    def change_agents_state(self, index_list: List[int], states: Union[MedicalState, Iterable[MedicalState]]) -> None:
        self.medical_state[index_list] = states

    def __getitem__(self, item):
        return self.at(item)

    # TODO: probably just get rid of it and use __getitem__ with '[]' instead
    def at(self, index_list):
        agents = AgentsDf()
        agents.state_machine = self.state_machine
        agents.age = self.age[index_list]
        agents.medical_state = self.medical_state[index_list]
        agents._ever_tested_positive = self._ever_tested_positive[index_list]
        agents._test_result = self._test_result[index_list]
        agents._date_of_last_test = self._date_of_last_test[index_list]
        agents.num_agents = 1 if isinstance(index_list, (int, np.integer)) else agents.medical_state.shape[0]
        agents.index = self.index[index_list]

        return agents

    def test_candidates(self, test_location: DetectionSettings, curr_date: int) -> Iterable[int]:
        def want_to_be_tested():
            random_vec = np.random.random(self.num_agents)
            return self.test_willingness > random_vec

        def alive():
            return self.medical_state != self.state_machine.states_by_name['Deceased']

        def can_be_tested_again():
            neg_gap = test_location.testing_gap_after_negative_test
            pos_gap = test_location.testing_gap_after_positive_test

            return (
                # never tested
                (self._test_result == DetectionResult.NOT_TAKEN) |
                # tested negative but gap passed
                ((self._test_result == DetectionResult.NEGATIVE) &
                 (self._date_of_last_test > curr_date - neg_gap)) |
                # tested positive but gap passed
                ((self._test_result == DetectionResult.POSITIVE) &
                 (self._date_of_last_test > curr_date - pos_gap))
            )

        return self.index[want_to_be_tested() & alive() & can_be_tested_again()]

    @property
    def contagiousness(self):
        return np.array([ms.contagiousness for ms in self.medical_state])

    @property
    def susceptible(self):
        return np.array([ms.susceptible for ms in self.medical_state])

    @property
    def detectable(self):
        return np.array([ms.detectable for ms in self.medical_state])

    @property
    def test_willingness(self):
        return np.array([ms.test_willingness for ms in self.medical_state])

    def set_tests_start(self, agent_index: Union[int, Iterable[int]], current_step: int):
        self._date_of_last_test[agent_index] = current_step

    def set_tests_results(self, agent_index: Union[int, Sequence[int]],
                          test_result: Union[DetectionResult, Iterable[DetectionResult]]):
        self._test_result[agent_index] = test_result
        self._ever_tested_positive[agent_index[test_result == DetectionResult.POSITIVE]] = True

    def ever_tested_positive(self, agent_index: Union[int, Iterable[int]]) -> Union[bool, Iterable[bool]]:
        return self._ever_tested_positive[agent_index]
