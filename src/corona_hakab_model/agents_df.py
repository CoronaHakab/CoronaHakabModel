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
                self._date_of_last_test, self.num_agents, self.index, self._contagiousness, self._susceptible, \
                self._detectable, self._test_willingness = [None] * 12
            return

        # initial data
        self.state_machine = state_machine
        initial: MedicalState = state_machine.initial

        num_agents = len(agents)

        self._ever_tested_positive = np.full(num_agents, False)
        self._test_result = np.full(num_agents, DetectionResult.NOT_TAKEN)
        self._date_of_last_test = np.full(num_agents, 0)

        self._contagiousness = np.full(num_agents, initial.contagiousness, dtype=np.float)
        self._susceptible = np.full(num_agents, initial.susceptible)
        self._detectable = np.full(num_agents, initial.detectable)
        self._test_willingness = np.full(num_agents, initial.test_willingness, dtype=np.float)

        self.age = np.array([a.age for a in agents])
        self.medical_state: np.ndarray = np.full(num_agents, initial)
        self.index = np.array([a.index for a in agents])
        self.num_agents = num_agents

    def __getitem__(self, items):
        agents = AgentsDf()
        agents._ever_tested_positive = self._ever_tested_positive[items]
        agents._test_result = self._test_result[items]
        agents._date_of_last_test = self._date_of_last_test[items]

        agents._contagiousness = self._contagiousness[items]
        agents._susceptible = self._susceptible[items]
        agents._detectable = self._detectable[items]
        agents._test_willingness = self._test_willingness[items]

        agents.state_machine = self.state_machine
        agents.age = self.age[items]
        agents.medical_state = self.medical_state[items]
        agents.index = self.index[items]
        agents.num_agents = 1 if isinstance(items, (int, np.integer)) else agents.medical_state.shape[0]

        return agents

    @property
    def contagiousness(self) -> np.ndarray:
        return self._contagiousness

    @property
    def susceptible(self) -> np.ndarray:
        return self._susceptible

    @property
    def detectable(self) -> np.ndarray:
        return self._detectable

    @property
    def test_willingness(self) -> np.ndarray:
        return self._test_willingness

    @property
    def ever_tested_positive(self) -> np.ndarray:
        return self._ever_tested_positive

    def n_agents(self) -> int:
        return self.num_agents

    def agents_indices(self) -> Iterable[int]:
        return self.index

    def change_agents_state(self, index_list: List[int], states: Union[MedicalState, Iterable[MedicalState]]) -> None:
        self.medical_state[index_list] = states

        if isinstance(states, MedicalState):
            states = [states]
        self._contagiousness[index_list] = [s.contagiousness for s in states]
        self._susceptible[index_list] = [s.susceptible for s in states]
        self._detectable[index_list] = [s.detectable for s in states]
        self._test_willingness[index_list] = [s.test_willingness for s in states]

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

        return self.index[alive() & want_to_be_tested() & can_be_tested_again()]

    def set_tests_start(self, agent_index: Union[int, Iterable[int]], current_step: int):
        self._date_of_last_test[agent_index] = current_step

    def set_tests_results(self, agent_index: Union[int, Sequence[int]],
                          test_result: Union[DetectionResult, Iterable[DetectionResult]]):
        self._test_result[agent_index] = test_result
        self._ever_tested_positive[agent_index[test_result == DetectionResult.POSITIVE]] = True
