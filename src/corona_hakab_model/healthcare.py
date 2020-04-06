from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Callable, List, Dict

import numpy as np
from agent import Agent
from util import Queue

if TYPE_CHECKING:
    from manager import SimulationManager


PendingTestResult = namedtuple("PendingTestResult", ["agent", "test_result", "original_duration"])


class PendingTestResults(Queue[PendingTestResult]):
    def __init__(self):
        super().__init__()


class DetectionTest:
    def __init__(self, detection_prob, false_alarm_prob, time_until_result):
        self.detection_prob = detection_prob
        self.false_alarm_prob = false_alarm_prob
        self.time_until_result = time_until_result

    def test(self, agent: Agent):
        if agent.medical_state.detectable:
            test_result = np.random.rand() < self.detection_prob
        else:
            test_result = np.random.rand() < self.false_alarm_prob

        pending_result = PendingTestResult(agent, test_result, self.time_until_result)
        return pending_result


class HealthcareManager:
    def __init__(
            self, sim_manager: SimulationManager,
            daily_num_of_test_schedule: Dict,
            detection_test: DetectionTest,
            testing_priorities: List[Callable[[Agent], bool]]
    ):
        self.testing_priorities = testing_priorities
        self.daily_num_of_test_schedule = daily_num_of_test_schedule
        self.detection_test = detection_test
        self.manager = sim_manager

        if 0 not in daily_num_of_test_schedule.keys():
            raise Exception("The initial number of tests (step=0) wasn't specified in the given schedule: "
                            f"{daily_num_of_test_schedule}")


    def _get_testable(self):
        tested_pos_too_recently = (
                self.manager.tested_vector
                & self.manager.tested_positive_vector
                & (
                        self.manager.current_step - self.manager.date_of_last_test
                        < self.manager.consts.testing_gap_after_positive_test
                )
        )

        tested_neg_too_recently = (
                self.manager.tested_vector
                & np.logical_not(self.manager.tested_positive_vector)
                & (
                        self.manager.current_step - self.manager.date_of_last_test
                        < self.manager.consts.testing_gap_after_negative_test
                )
        )

        return np.logical_not(tested_pos_too_recently | tested_neg_too_recently) & self.manager.living_agents_vector

    def _get_current_num_of_tests(self, current_step):
        keys = list(self.daily_num_of_test_schedule.keys())
        closest_key = max([i for i in self.daily_num_of_test_schedule.keys() if i <= current_step])
        return self.daily_num_of_test_schedule[closest_key]

    def testing_step(self):
        num_of_tests = self._get_current_num_of_tests(self.manager.current_step)

        # Who can to be tested
        want_to_be_tested = np.random.random(len(self.manager.agents)) < self.manager.test_willingness_vector
        can_be_tested = self._get_testable()
        test_candidates = want_to_be_tested & can_be_tested

        test_candidates_inds = set(np.flatnonzero(test_candidates))
        tested: List[PendingTestResult] = []

        if len(test_candidates_inds) < num_of_tests:
            # There are more tests than candidates. Don't check the priorities
            for ind in test_candidates_inds:
                tested.append(self.detection_test.test(self.manager.agents[ind]))
                num_of_tests -= 1
        else:
            for priority_lambda in list(self.testing_priorities):
                # First test the prioritized candidates
                for ind in np.random.permutation(list(test_candidates_inds)):
                    # permute the indices so we won't always test the lower indices
                    if priority_lambda(self.manager.agents[ind]):
                        tested.append(self.detection_test.test(self.manager.agents[ind]))
                        test_candidates_inds.remove(ind)  # Remove so it won't be tested again
                        num_of_tests -= 1

                        if num_of_tests == 0:
                            return tested

            # Test the low prioritized now
            num_of_low_priority_to_test = min(num_of_tests, len(test_candidates_inds))
            low_priority_tested = [
                self.detection_test.test(self.manager.agents[ind])
                for ind in np.random.permutation(list(test_candidates_inds))[:num_of_low_priority_to_test]
            ]
            tested += low_priority_tested
            num_of_tests -= len(low_priority_tested)

        # There are some tests left. Choose randomly from outside the pool
        test_leftovers_candidates_inds = np.flatnonzero(can_be_tested & np.logical_not(want_to_be_tested))
        will_be_tested_inds = np.random.choice(
            test_leftovers_candidates_inds, min(num_of_tests, len(test_leftovers_candidates_inds)), replace=False
        )

        for ind in will_be_tested_inds:
            tested.append(self.detection_test.test(self.manager.agents[ind]))
            num_of_tests -= 1

        return tested
