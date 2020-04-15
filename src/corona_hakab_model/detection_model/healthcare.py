from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, NamedTuple

import numpy as np
import pandas as pd

from detection_model.detection_testing_types import DetectionSettings
from util import Queue

if TYPE_CHECKING:
    from manager import SimulationManager


class PendingTestResult(NamedTuple):
    agent_index: int
    test_result: DetectionResult
    original_duration: int

    def duration(self):
        return self.original_duration


class PendingTestResults(Queue[PendingTestResult]):
    pass


class DetectionTest:
    def __init__(self, detection_prob, false_alarm_prob, time_until_result):
        self.detection_prob = detection_prob
        self.false_alarm_prob = false_alarm_prob
        self.time_until_result = time_until_result

    def test(self, agents, agents_indices):
        rand_vec = np.random.random(len(agents_indices))
        # is_detectable_mask = np.array([ms.detectable for ms in agents.medical_state])
        is_detectable_mask = agents.detectable.values
        test_results = (
                (is_detectable_mask & (rand_vec < self.detection_prob)) |
                (~is_detectable_mask & (rand_vec < self.false_alarm_prob))
        )

        test_results = [DetectionResult.POSITIVE if test_result else DetectionResult.NEGATIVE for test_result in
                        test_results]
        pending_results = [PendingTestResult(agent_ind, test_result, self.time_until_result) for agent_ind, test_result
                           in zip(agents_indices, test_results)]
        return pending_results


class HealthcareManager:
    def __init__(self, sim_manager: SimulationManager):
        self.manager = sim_manager
        for testing_location in self.manager.consts.detection_pool:
            if 0 not in testing_location.daily_num_of_tests_schedule.keys():
                raise Exception(
                    "The initial number of tests (step=0) wasn't specified in the given schedule: "
                    f"{self.manager.consts.daily_num_of_test_schedule}"
                )

    def _get_current_num_of_tests(self, current_step, test_location: DetectionSettings):
        closest_key = max([i for i in test_location.daily_num_of_tests_schedule.keys() if i <= current_step])
        return test_location.daily_num_of_tests_schedule[closest_key]

    def testing_step(self):
        tested: List[PendingTestResult] = []
        # TODO: vectorize on test locations?
        for test_location in self.manager.consts.detection_pool:
            num_of_tests = self._get_current_num_of_tests(self.manager.current_step, test_location)

            # Who can be tested
            test_candidates_inds = self.manager.agents_df.test_candidates(test_location, self.manager.current_step)
            test_candidates_inds -= set(result.agent_index for result in tested)

            if len(test_candidates_inds) < num_of_tests:
                # There are more tests than candidates. Don't check the priorities
                tested += test_location.detection_test.test(self.manager.agents_df.at(test_candidates_inds),
                                                            test_candidates_inds)
                num_of_tests -= len(test_candidates_inds)
                # for ind in test_candidates_inds:
                #     tested.append(test_location.detection_test.test(self.manager.agents_df.at(ind), ind))
                #     num_of_tests -= 1
            else:
                num_of_tests = self._test_according_to_priority(num_of_tests, test_candidates_inds, test_location,
                                                                tested)

                # Test the low prioritized now
                num_of_low_priority_to_test = min(num_of_tests, len(test_candidates_inds))
                if num_of_low_priority_to_test > 0:
                    low_priority_indices = np.random.permutation(list(test_candidates_inds))[
                                           :num_of_low_priority_to_test]
                    low_priority_tested = test_location.detection_test.test(
                        self.manager.agents_df.at(low_priority_indices),
                        low_priority_indices)

                    # low_priority_tested = [
                    #     test_location.detection_test.test(self.manager.agents_df.at(ind), ind)
                    #     for ind in np.random.permutation(list(test_candidates_inds))[:num_of_low_priority_to_test]
                    # ]
                    tested += low_priority_tested
                    num_of_tests -= len(low_priority_tested)

        # # There are some tests left. Choose randomly from outside the pool
        # test_leftovers_candidates_inds = np.flatnonzero(can_be_tested & np.logical_not(want_to_be_tested))
        # will_be_tested_inds = np.random.choice(
        #     test_leftovers_candidates_inds, min(num_of_tests, len(test_leftovers_candidates_inds)), replace=False
        # )
        #
        # for ind in will_be_tested_inds:
        #     tested.append(self.manager.consts.detection_test.test(self.manager.agents[ind]))
        #     num_of_tests -= 1
        #
        return tested

    def _test_according_to_priority(self, num_of_tests, test_candidates_inds, test_location, tested):
        candidates_by_priority_inds = []
        test_candidates: pd.Dataframe = self.manager.agents_df.at(test_candidates_inds)

        # First test the prioritized candidates
        for detection_priority in list(test_location.testing_priorities):
            is_priority_agent = {agent_index: detection_priority.is_agent_prioritized(agent) for agent_index, agent in
                                 zip(test_candidates_inds, test_candidates.itertuples())}
            # permute the indices so we won't always test the lower indices
            for ind in np.random.permutation(list(test_candidates_inds)):
                if is_priority_agent[ind]:
                    # tested.append(test_location.detection_test.test(self.manager.agents_df.at(ind), ind))
                    candidates_by_priority_inds.append(ind)
                    test_candidates.drop(ind, inplace=True)
                    test_candidates_inds.remove(ind)  # Remove so it won't be tested again
                    num_of_tests -= 1
                    # need no more candidates to test
                    if num_of_tests == 0:
                        break
            # need no more candidates to test
            if num_of_tests == 0:
                break

        if candidates_by_priority_inds:
            tested += test_location.detection_test.test(self.manager.agents_df.at(candidates_by_priority_inds),
                                                        candidates_by_priority_inds)
        return num_of_tests


class DetectionResult(Enum):
    NOT_TAKEN = 0
    POSITIVE = 1
    NEGATIVE = 2
