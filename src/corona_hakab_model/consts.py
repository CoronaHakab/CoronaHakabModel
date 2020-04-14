import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from medical_state_consts.detection_test import DetectionTestConsts

from generation.connection_types import ConnectionTypes
from numpy.random import random
from policies_manager import ConditionedPolicy, Policy
# todo make sure we only use this
generator = np.random.default_rng()

"""
Overview:

Consts class is a named tuple holding all important consts for the simulation stage.
it may either be made using default params, or by loading parameters from a file.
Usage:
1. Create a default consts object - consts = Consts()
2. Load a parameters file - consts = Consts.from_file(path)
"""


# TODO split into a couple of files. one for each aspect of the simulation
@dataclass
class Consts:
    # attributes and default values:
    total_steps: int = 350
    initial_infected_count: int = 20

    # base r0 of the disease
    r0: float = 2.4

    # --Detection tests params--
    # the probability that an infected agent is asking to be tested
    detection_test_consts = DetectionTestConsts()

    @property
    def detection_test(self):
        return self.detection_test_consts.detection_test

    # --policies params--
    change_policies: bool = False

    # a dictionary of day:([ConnectionTypes], message). on each day, keeps only the given connection types opened
    policy_changes: Dict[int, tuple] = field(default_factory=lambda: {
        40: ([ConnectionTypes.Family, ConnectionTypes.Other], "closing schools and work sites"),
        70: ([ConnectionTypes.Family, ConnectionTypes.Other, ConnectionTypes.School], "opening schools"),
        100: (ConnectionTypes, "opening work sites"),
    })
    # policies acting on a specific connection type, when a term is satisfied
    partial_opening_active: bool = True

    # each connection type gets a list of conditioned policies.
    # each conditioned policy actives a specific policy when a condition is satisfied.
    # each policy changes the multiplication factor of a specific circle.
    # each policy is activated only if a list of terms is fulfilled.
    connection_type_to_conditioned_policy: Dict[ConnectionTypes, List[ConditionedPolicy]] = field(
        default_factory=lambda: {
            ConnectionTypes.School: [
                ConditionedPolicy(
                    activating_condition=lambda kwargs: len(
                        np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
                    policy=Policy(0, [lambda circle: random() > 0]),
                    message="closing all schools",
                ),
                ConditionedPolicy(
                    activating_condition=lambda kwargs: len(
                        np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
                    policy=Policy(1, [lambda circle: random() > 1]),
                    active=True,
                    message="opening all schools",
                ),
            ],
            ConnectionTypes.Work: [
                ConditionedPolicy(
                    activating_condition=lambda kwargs: len(
                        np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
                    policy=Policy(0, [lambda circle: random() > 0]),
                    message="closing all workplaces",
                ),
                ConditionedPolicy(
                    activating_condition=lambda kwargs: len(
                        np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
                    policy=Policy(0, [lambda circle: random() > 1]),
                    active=True,
                    message="opening all workplaces",
                ),
            ],
        })

    @classmethod
    def json_dict_to_instance(cls, **kwargs):
        detection_test_consts = kwargs.pop('detection_test_consts')
        policy_changes = kwargs.pop('policy_changes')
        self = cls(**kwargs)
        self.detection_test_consts = cls.detection_test_consts.json_dict_to_instance(**detection_test_consts)
        self.policy_changes = {
            days: (map(lambda ct: ConnectionTypes[ct], data[0]),
                   data[1],) for days, data in policy_changes.items()
        }
        return self

    @classmethod
    def from_json(cls, param_path):
        """
        Load parameters from JSON file and return Consts object with those values.

        Documentation about what is allowed and not allowed can be found at the top of this page.
        """
        with open(param_path, "rt") as read_file:
            data = read_file.read()

        return Consts.json_dict_to_instance(**json.loads(data))

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__
