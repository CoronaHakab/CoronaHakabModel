from __future__ import annotations

from typing import TYPE_CHECKING, List
import numpy as np

from common.detection_testing_types import DetectionSettings, PendingTestResult, PendingTestResults

if TYPE_CHECKING:
    from manager import SimulationManager


def _get_current_num_of_tests(current_step, test_location: DetectionSettings):
    closest_key = max([i for i in test_location.daily_num_of_tests_schedule.keys() if i <= current_step])
    return test_location.daily_num_of_tests_schedule[closest_key]


class HealthcareManager:
    __slots__ = ("manager", "positive_detected_today", "freed_neg_tested",
                 "pending_test_results", "num_of_tested")

    def __init__(self, sim_manager: SimulationManager):
        self.manager = sim_manager
        for testing_location in self.manager.consts.detection_pool:
            if 0 not in testing_location.daily_num_of_tests_schedule.keys():
                raise Exception(
                    "The initial number of tests (step=0) wasn't specified in the given schedule: "
                    f"{self.manager.consts.daily_num_of_test_schedule}"
                )
        self.positive_detected_today = set()
        self.freed_neg_tested = set()
        self.pending_test_results = PendingTestResults()
        self.num_of_tested = None

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

    def step(self):
        self.manager.left_isolation_by_reason.clear()
        self.freed_neg_tested.clear()
        self.progress_tests(self.testing_step())
        # TODO: Move isolation functions to here
        if self.manager.consts.day_to_start_isolations <= self.manager.current_step:
            self.manager.progress_isolations()

    def testing_step(self):
        want_to_be_tested = np.random.random(len(self.manager.agents)) < self.manager.test_willingness_vector
        tested: List[PendingTestResult] = []

        for test_location in self.manager.consts.detection_pool:
            num_of_tests = _get_current_num_of_tests(self.manager.current_step, test_location)

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
        self.num_of_tested = len(tested)
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

    def progress_tests(self, new_tests: List[PendingTestResult]):
        self.positive_detected_today.clear()
        new_results = self.pending_test_results.advance()
        for agent, test_result, _ in new_results:
            if test_result:
                self.positive_detected_today.add(agent.index)
                if self.manager.consecutive_negative_tests[agent.index] > 0:
                    self.manager.consecutive_negative_tests[agent.index] = 0
            else:
                # When isolated agent gets negative result, free him NOW!
                if self.manager.consecutive_negative_tests[agent.index] == \
                        self.manager.consts.num_test_to_exit_isolation:
                    self.manager.step_to_free_agent[agent.index] = self.manager.current_step
                    self.freed_neg_tested.add(agent.index)
            agent.set_test_result(test_result)

        for new_test in new_tests:
            new_test.agent.set_test_start()
            self.pending_test_results.append(new_test)
