from dataclasses import dataclass, field
from typing import Dict

from healthcare import DetectionTest


@dataclass
class DetectionTestConsts:
    detection_test: DetectionTest = DetectionTest(detection_prob=0.98, false_alarm_prob=0.02, time_until_result=3)
    daily_num_of_tests_schedule: Dict = field(default_factory=lambda: {0: 100, 10: 1000, 20: 2000, 50: 5000})
    testing_gap_after_positive_test: int = 10
    testing_gap_after_negative_test: int = 5
    testing_priorities: tuple = ("Symptomatic", "Recovered", )

    @classmethod
    def json_dict_to_instance(cls, **kwargs):
        self = cls(**kwargs)
        self.detection_test = DetectionTest(
            detection_prob=kwargs['detection_test']['detection_prob'],
            false_alarm_prob=kwargs['detection_test']['false_alarm_prob'],
            time_until_result=kwargs['detection_test']['time_until_result']
        )
        self.testing_priorities = tuple(kwargs['testing_priorities'])
        return self
