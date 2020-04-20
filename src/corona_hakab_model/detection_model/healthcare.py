from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Dict

import numpy as np

from agent import Agent
from detection_model.detection_testing_types import DetectionSettings
from medical_state import MedicalState
from util import Queue

if TYPE_CHECKING:
    from manager import SimulationManager


class PendingTestResult(NamedTuple):
    agent: Agent
    test_result: bool
    original_duration: int

    def duration(self):
        return self.original_duration


class PendingTestResults(Queue[PendingTestResult]):
    pass


class DetectionTest:
    def __init__(self, state_to_detection_prop: Dict[MedicalState, float], time_until_result):
        self.state_to_detection_prop = state_to_detection_prop
        self.time_until_result = time_until_result

    def test(self, agent: Agent):
        detection_prob = self.state_to_detection_prop[agent.medical_state.name]
        test_result = np.random.rand() < detection_prob
        pending_result = PendingTestResult(agent, test_result, self.time_until_result)
        return pending_result


class HealthcareManager:
    def __init__(self, sim_manager: SimulationManager):
        self.manager = sim_manager
        for testing_location in self.manager.consts.detection_pool:
            if 0 not in testing_location.daily_num_of_tests_schedule.keys():
                raise Exception(
                    "The initial number of tests (step=0) wasn't specified in the given schedule: "
                    f"{self.manager.consts.daily_num_of_test_schedule}"
                )

    def _get_testable(self, test_location: DetectionSettings):
        tested_pos_too_recently = (
                self.manager.tested_vector
                & self.manager.tested_positive_vector
                & (
                        self.manager.current_step - self.manager.date_of_last_test
                        < test_location.testing_gap_after_positive_test
                )
        )

        tested_neg_too_recently = (
                self.manager.tested_vector
                & np.logical_not(self.manager.tested_positive_vector)
                & (
                        self.manager.current_step - self.manager.date_of_last_test
                        < test_location.testing_gap_after_negative_test
                )
        )

        return np.logical_not(tested_pos_too_recently | tested_neg_too_recently) & self.manager.living_agents_vector

    def _get_current_num_of_tests(self, current_step, test_location: DetectionSettings):
        closest_key = max([i for i in test_location.daily_num_of_tests_schedule.keys() if i <= current_step])
        return test_location.daily_num_of_tests_schedule[closest_key]

    def testing_step(self):
        want_to_be_tested = np.random.random(len(self.manager.agents)) < self.manager.test_willingness_vector
        tested: List[PendingTestResult] = []

        for test_location in self.manager.consts.detection_pool:
            num_of_tests = self._get_current_num_of_tests(self.manager.current_step, test_location)

            # Who can to be tested
            can_be_tested = self._get_testable(test_location)
            test_candidates = want_to_be_tested & can_be_tested
            test_candidates_inds = set(np.flatnonzero(test_candidates))
            test_candidates_inds -= set(result.agent.index for result in tested)

            if len(test_candidates_inds) < num_of_tests:
                # There are more tests than candidates. Don't check the priorities
                for ind in test_candidates_inds:
                    tested.append(test_location.detection_test.test(self.manager.agents[ind]))
                    num_of_tests -= 1
            else:
                self._test_according_to_priority(num_of_tests, test_candidates_inds, test_location,
                                                 tested)

        return tested

    def _test_according_to_priority(self, num_of_tests, test_candidates_inds, test_location, tested):
        for detection_priority in list(test_location.testing_priorities):
            # First test the prioritized candidates
            for ind in np.random.permutation(list(test_candidates_inds)):
                # permute the indices so we won't always test the lower indices
                if detection_priority.is_agent_prioritized(self.manager.agents[ind]):
                    tested.append(test_location.detection_test.test(self.manager.agents[ind]))
                    test_candidates_inds.remove(ind)  # Remove so it won't be tested again
                    num_of_tests -= 1

                    if num_of_tests == 0:
                        return 0
        return num_of_tests
