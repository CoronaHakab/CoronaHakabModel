from __future__ import annotations

from collections import Callable
from dataclasses import dataclass
from typing import Dict, List
from typing import TYPE_CHECKING

from numpy import inf

from common.agent import Agent

if TYPE_CHECKING:
    from detection_model.healthcare import DetectionTest


class DetectionPriority:

    def __init__(self, func: Callable, max_tests: int = inf):
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
