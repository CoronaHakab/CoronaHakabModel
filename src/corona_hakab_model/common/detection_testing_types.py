from __future__ import annotations
from collections import Callable
from dataclasses import dataclass
from typing import Dict, List, NamedTuple
from copy import deepcopy
from numpy import inf
import numpy as np

from .agent import Agent
from .medical_state import MedicalState
from .util import Queue


class PendingTestResult(NamedTuple):
    agent: Agent
    test_result: bool
    original_duration: int

    def duration(self):
        return self.original_duration


class PendingTestResults(Queue[PendingTestResult]):
    pass


class DetectionTest:
    def __init__(self,
                 state_to_detection_prop: Dict[MedicalState, float],
                 time_dist_until_result):
        self.state_to_detection_prop = deepcopy(state_to_detection_prop)
        self.time_dist_until_result = time_dist_until_result

    def get_times_to_get_results(self, number_of_agents=1):
        if number_of_agents == 1:
            return self.time_dist_until_result()
        return self.time_dist_until_result(size=number_of_agents)

    def test(self, agent: Agent):
        detection_prob = self.state_to_detection_prop[agent.medical_state.name]
        test_result = np.random.rand() < detection_prob
        time_to_result = self.get_times_to_get_results()
        pending_result = PendingTestResult(agent,
                                           test_result,
                                           time_to_result)
        return pending_result


class DetectionPriority:
    def __init__(self,
                 func: Callable,
                 max_tests: int = inf):
        """
        medical test for infection detection.
        supports max number of tests for its kind. if not specified, will be infinity.
        @param func: callable lambda or function that receives an agent and returns if should be tested.
                    should return bool.
        @param max_tests: maximum number of tests allowed.
        """

        self.func = func
        self.max_tests = max_tests
        self.count = 0

    def is_agent_prioritized(self, agent: Agent) -> bool:
        """
        activating the function on an agent. if exceeded the allowed number of tests, return false.
        else: increase the counter and activate the function, return it's outcome.
        """
        if self.count >= self.max_tests:
            return False
        self.count += 1
        return self.func(agent)


@dataclass()
class DetectionSettings:
    name: str
    detection_test: DetectionTest
    daily_num_of_tests_schedule: Dict[int, int]  # day -> n_tests
    testing_gap_after_positive_test: int  # days
    testing_gap_after_negative_test: int  # days
    testing_priorities: List[DetectionPriority]
