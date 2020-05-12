import os
from functools import lru_cache
from typing import Dict, List, NamedTuple, Union, Callable
import jsonpickle
import numpy as np
from numpy.random import random

from common.detection_testing_types import DetectionSettings, DetectionPriority, DetectionTest
from generation.connection_types import ConnectionTypes
from common.medical_state import ContagiousState, ImmuneState, SusceptibleState
from common.medical_state_machine import MedicalStateMachine
from policies_manager import ConditionedPolicy, Policy
from common.state_machine import StochasticState, TerminalState
from common.util import dist, BucketDict

TransitionProbType = BucketDict[int, Union[float, type(...)]]

"""
Overview:

Consts class is a named tuple holding all important consts for the simulation stage.
it may either be made using default params, or by loading parameters from a file.
Usage:
1. Create a default consts object - consts = Consts()
2. Load a parameters file - consts = Consts.from_file(path)
"""


# TODO split into a couple of files. one for each aspect of the simulation
class Consts(NamedTuple):
    # medical states
    LATENT: str = "Latent"
    SUSCEPTIBLE: str = "Susceptible"
    RECOVERED: str = "Recovered"
    DECEASED: str = "Deceased"
    PRE_RECOVERED: str = "PreRecovered"
    IMPROVING_HEALTH: str = "ImprovingHealth"
    NEED_ICU: str = "NeedICU"
    NEED_OF_CLOSE_MEDICAL_CARE: str = "NeedOfCloseMedicalCare"
    MILD_CONDITION: str = "Mild-Condition"
    PRE_SYMPTOMATIC: str = "Pre-Symptomatic"
    ASYMPTOMATIC: str = "Asymptomatic"
    # attributes and default values:

    total_steps: int = 150
    initial_infected_count: int = 20
    export_infected_agents_interval: int = 1000

    # Size of population to estimate expected time for each state
    population_size_for_state_machine_analysis: int = 25_000

    # Backtrack infection sources?
    backtrack_infection_sources: bool = False

    # Tsvika: Currently the distribution is selected based on the number of input parameters.
    # Think we should do something more readable later on.
    # For example: "latent_presymp_to_pre_symptomatic_days": {"type":"uniform","lower_bound":1,"upper_bound":3}
    # disease states transition lengths distributions

    # Binomial distribution for all ages
    latent_to_pre_symptomatic_days: BucketDict[int, Callable] = BucketDict({0: dist(
        list(range(1, 14)),
        [0.083, 0.13325, 0.16925, 0.169, 0.144, 0.10675, 0.0725, 0.04675, 0.02925, 0.021, 0.01275, 0.00775, 0.00475]
    )})

    latent_to_asymptomatic_days: BucketDict[int, Callable] = latent_to_pre_symptomatic_days

    pre_symptomatic_to_mild_days: BucketDict[int, Callable] = BucketDict({0: dist(
        [(0, 3), (1, 2), (2, 1), (3, 0)],  # Instead of days, each element in the array is (infection offset, n_days)
        [0.25] * 4)
    })

    #
    mild_to_close_medical_care_days: BucketDict[int, Callable] = BucketDict({0: dist(
        list(range(1, 13)),
        [0, 0, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.01]
    )})

    #
    mild_to_need_icu_days: BucketDict[int, Callable] = BucketDict({0: dist(
        list(range(1, 31)),
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.012, 0.019, 0.032, 0.046, 0.059, 0.069, 0.075, 0.077, 0.075, 0.072, 0.066, \
         0.060, 0.053, 0.046, 0.040, 0.035, 0.030, 0.028, 0.026, 0.022, 0.020, 0.015, 0.010, 0.010, 0.000]
    )})

    #
    mild_to_pre_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(
        list(range(1, 29)),
        [0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.004, 0.008, 0.013, 0.022, 0.032, 0.046, 0.06, 0.075, 0.088, 0.097, \
         0.1, 0.097, 0.088, 0.075, 0.06, 0.046, 0.032, 0.022, 0.013, 0.008, 0.004, 0.002]
    )})

    #
    close_medical_care_to_icu_days: BucketDict[int, Callable] = BucketDict({0: dist(10, 12, 14)})

    #
    close_medical_care_to_mild_days: BucketDict[int, Callable] = BucketDict({0: dist(8, 10, 12)})

    #
    need_icu_to_deceased_days: BucketDict[int, Callable] = BucketDict({0: dist(
        list(range(1, 21)),
        [0.030, 0.100, 0.120, 0.110, 0.090, 0.080, 0.075, 0.070, 0.065, 0.050, 0.040, 0.035, 0.030, 0.025, 0.020, 0.015, \
         0.012, 0.010, 0.008, 0.005]
    )})

    #
    need_icu_to_improving_days: BucketDict[int, Callable] = BucketDict({0: dist(
        list(range(1, 26)),
        [0.020, 0.040, 0.080, 0.100, 0.100, 0.080, 0.070, 0.065, 0.060, 0.055, 0.045, 0.040, 0.038, 0.032, 0.030, 0.025, \
         0.020, 0.015, 0.012, 0.012, 0.010, 0.010, 0.008, 0.005, 0.005]
    )})

    #
    improving_to_need_icu_days: BucketDict[int, Callable] = BucketDict({0: dist(21, 42)})

    #
    improving_to_pre_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(21, 42)})  # TODO: check why so long

    #
    improving_to_mild_condition_days: BucketDict[int, Callable] = BucketDict({0: dist(21, 42)})

    #
    pre_recovered_to_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(
        [14, 28],
        [0.8, 0.2]
    )})

    #
    asymptomatic_to_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(
        list(range(1, 36)),
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.013, 0.016, 0.025, 0.035, 0.045, 0.054, 0.062, \
         0.066, 0.069, 0.069, 0.066, 0.063, 0.058, 0.053, 0.056, 0.041, 0.040, 0.033, 0.030, 0.025, 0.020, 0.015, 0.015, 0.015, 0.010, 0.010]
    )})

    # state machine transfer probabilities
    # probability of '...' equals (1 - all other transfers)
    # it should always come last after all other transition probabilities were defined
    latent_to_asymptomatic_prob: TransitionProbType = BucketDict({0: 0.3})
    latent_to_pre_symptomatic_prob:  TransitionProbType = BucketDict({0: ...})
    pre_symptomatic_to_mild_prob:  TransitionProbType = BucketDict({0: ...})
    mild_to_close_medical_care_prob:  TransitionProbType = BucketDict({0: 0.2375})
    mild_to_need_icu_prob:  TransitionProbType = BucketDict({0: 0.0324})
    mild_to_pre_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    close_medical_care_to_icu_prob:  TransitionProbType = BucketDict({0: 0.26})
    close_medical_care_to_mild_prob:  TransitionProbType = BucketDict({0: ...})
    need_icu_to_deceased_prob:  TransitionProbType = BucketDict({0: 0.3})
    need_icu_to_improving_prob:  TransitionProbType = BucketDict({0: ...})
    improving_to_need_icu_prob:  TransitionProbType = BucketDict({0: 0})
    improving_to_pre_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    improving_to_mild_condition_prob:  TransitionProbType = BucketDict({0: 0})
    pre_recovered_to_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    asymptomatic_to_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    # infections ratios, See bucket dict for more info on how to use.
    pre_symptomatic_infection_ratio: BucketDict[int, int] = BucketDict({0: [0.14, 0.86, 1]})  # if x greater than biggest key, x is biggest key
    asymptomatic_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0.14, 0.86, 1, 0.82, 0.59, 0.41, 0.27, 0.18, 0.14, 0.09, 0.05]})
    mild_condition_infection_ratio: BucketDict[int, int] = BucketDict({0: [0.82, 0.59, 0.41, 0.27, 0.18, 0.14, 0.09, 0.05]})
    latent_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    need_close_medical_care_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    need_icu_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    improving_health_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    pre_recovered_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    # base r0 of the disease
    r0: float = 2.4

    # --Detection tests params--
    # the probability that an infected agent is asking to be tested
    susceptible_test_willingness: float = 0.01
    latent_test_willingness: float = 0.01
    asymptomatic_test_willingness: float = 0.01
    pre_symptomatic_test_willingness: float = 0.01
    mild_condition_test_willingness: float = 0.6
    need_close_medical_care_test_willingness: float = 0.9
    need_icu_test_willingness: float = 1.0
    improving_health_test_willingness: float = 1.0
    pre_recovered_test_willingness: float = 0.5
    recovered_test_willingness: float = 0.1
    detection_pool: List[DetectionSettings] = [
        DetectionSettings(
            name="hospital",
            detection_test=DetectionTest({
                SUSCEPTIBLE: 0.,
                LATENT: .98,
                RECOVERED: 0.,
                DECEASED: 0.,
                PRE_RECOVERED: .98,
                IMPROVING_HEALTH: .98,
                NEED_ICU: .98,
                NEED_OF_CLOSE_MEDICAL_CARE: .98,
                MILD_CONDITION: .98,
                PRE_SYMPTOMATIC: .98,
                ASYMPTOMATIC: .98,
            }, time_dist_until_result=dist(3)),  # Constant distribution
            daily_num_of_tests_schedule={0: 100, 10: 1000, 20: 2000, 50: 5000},
            testing_gap_after_positive_test=2,
            testing_gap_after_negative_test=1,
            testing_priorities=[
                DetectionPriority(
                    lambda agent: (agent.medical_state.name == "Symptomatic" and
                                   agent not in agent.manager.tested_positive_vector),
                    max_tests=100),
                DetectionPriority(
                    lambda agent: agent.medical_state.name == "Recovered"),
            ]),

        DetectionSettings(
            name="street",
            detection_test=DetectionTest({
                SUSCEPTIBLE: 0.,
                LATENT: .92,
                RECOVERED: 0.,
                DECEASED: 0.,
                PRE_RECOVERED: .92,
                IMPROVING_HEALTH: .92,
                NEED_ICU: .92,
                NEED_OF_CLOSE_MEDICAL_CARE: .92,
                MILD_CONDITION: .92,
                PRE_SYMPTOMATIC: .92,
                ASYMPTOMATIC: .92,
            }, time_dist_until_result=dist(5)),  # Constant distribution
            daily_num_of_tests_schedule={0: 500, 10: 1500, 20: 2500, 50: 7000},
            testing_gap_after_positive_test=3,
            testing_gap_after_negative_test=1,
            testing_priorities=[
                DetectionPriority(
                    lambda agent: agent.medical_state.name == "Symptomatic"),
                DetectionPriority(
                    lambda agent: agent.medical_state.name == "Recovered"),
            ]),
    ]
    should_isolate_positive_detected: bool = False
    isolate_after_num_day: int = 1  # will be in isolation the next day.
    p_will_obey_isolation: float = 1.0  # 100% will obey the isolation.
    isolation_factor: float = 0.0  # reduce agent's relations strength by a factor.

    # --policies params--
    change_policies: bool = False
    # a dictionary of day:([ConnectionTypes], message). on each day, keeps only the given connection types opened
    policies_changes: Dict[int, tuple] = {
        40: ([ConnectionTypes.Family, ConnectionTypes.Other], "closing schools, kindergartens and works"),
        70: ([ConnectionTypes.Family, ConnectionTypes.Other, ConnectionTypes.School, ConnectionTypes.Kindergarten],
             "opening schools and kindergartens"),
        100: (ConnectionTypes, "opening works"),
    }
    # policies acting on a specific connection type, when a term is satisfied
    partial_opening_active: bool = False
    # each connection type gets a list of conditioned policies.
    # each conditioned policy actives a specific policy when a condition is satisfied.
    # each policy changes the multiplication factor of a specific circle.
    # each policy is activated only if a list of terms is fulfilled.
    connection_type_to_conditioned_policy: Dict[ConnectionTypes, List[ConditionedPolicy]] = {
        ConnectionTypes.School: [
            ConditionedPolicy(
                activating_condition=lambda kwargs: np.count_nonzero(kwargs["manager"].contagiousness_vector > 0) > 1000,
                policy=Policy(0, [lambda circle: True]),
                message="closing all schools",
            ),
            ConditionedPolicy(
                activating_condition=lambda kwargs: np.count_nonzero(kwargs["manager"].contagiousness_vector > 0) < 500,
                policy=Policy(1, [lambda circle: False]),
                active=True,
                message="opening all schools",
            ),
        ],
        ConnectionTypes.Kindergarten: [
            ConditionedPolicy(
                activating_condition=lambda kwargs: np.count_nonzero(kwargs["manager"].contagiousness_vector > 0) > 1000,
                policy=Policy(0, [lambda circle: True]),
                message="closing all kindergartens",
            ),
            ConditionedPolicy(
                activating_condition=lambda kwargs: np.count_nonzero(kwargs["manager"].contagiousness_vector > 0) < 500,
                policy=Policy(1, [lambda circle: False]),
                active=True,
                message="opening all kindergartens",
            ),
        ],
        ConnectionTypes.Work: [
            ConditionedPolicy(
                activating_condition=lambda kwargs: np.count_nonzero(kwargs["manager"].contagiousness_vector > 0) > 1000,
                policy=Policy(0, [lambda circle: True]),
                message="closing all workplaces",
            ),
            ConditionedPolicy(
                activating_condition=lambda kwargs: np.count_nonzero(kwargs["manager"].contagiousness_vector > 0) < 500,
                policy=Policy(0, [lambda circle: False]),
                active=True,
                message="opening all workplaces",
            ),
        ],
    }

    @classmethod
    def from_file(cls, param_path):
        """
        Load parameters from file and return Consts object with those values.

        No need to sanitize the eval'd data as we disabled __builtins__ and only passed specific functions
        Documentation about what is allowed and not allowed can be found at the top of this page.
        """
        with open(param_path, "rt") as read_file:
            data = read_file.read()

        # expressions to evaluate
        expressions = {
            "__builtins__": None,
            "dist": dist,
            "DetectionSettings": DetectionSettings,
            "DetectionPriority": DetectionPriority,
            "DetectionTest": DetectionTest,
            "ConditionedPolicy": ConditionedPolicy,
            "ConnectionTypes": ConnectionTypes,
            "Policy": Policy,
            "random": random,
            "np": np,
            "BucketDict": BucketDict,
            "len": len,
        }

        parameters = eval(data, expressions)

        return cls(**parameters)

    def export(self, export_path, file_name: str = "simulation_consts.json"):
        if not file_name.endswith(".json"):
            file_name += ".json"
        with open(os.path.join(export_path, file_name), "w") as export_file:
            export_file.write(jsonpickle.encode(self._asdict()))

    @lru_cache(None)
    def medical_state_machine(self) -> MedicalStateMachine:
        class SusceptibleTerminalState(SusceptibleState, TerminalState):
            pass

        class ImmuneStochasticState(ImmuneState, StochasticState):
            pass

        class ContagiousStochasticState(ContagiousState, StochasticState):
            pass

        class ImmuneTerminalState(ImmuneState, TerminalState):
            pass

        # In order to deal with the latent + pre-symptomatic probabilities, we split latent state in two:
        # latent-presymp - which will have the duration of the incubation period (minus 1 day)
        # Presymp will have a short distribution of 1-3 days
        # latent-asymp - will have the durations of latent, summed by each day
        # probability for each is same as probability from latent to presymp and asymp

        susceptible = SusceptibleTerminalState(self.SUSCEPTIBLE, test_willingness=self.susceptible_test_willingness)
        latent = ContagiousStochasticState(
            self.LATENT,
            detectable=False,
            contagiousness=self.latent_infection_ratio,
            test_willingness=self.latent_test_willingness)
        asymptomatic = ContagiousStochasticState(
            self.ASYMPTOMATIC,
            detectable=True,
            contagiousness=self.asymptomatic_infection_ratio,
            test_willingness=self.asymptomatic_test_willingness
        )
        pre_symptomatic = ContagiousStochasticState(
            self.PRE_SYMPTOMATIC,
            detectable=True,
            contagiousness=self.pre_symptomatic_infection_ratio,
            test_willingness=self.pre_symptomatic_test_willingness,
        )
        mild_condition = ContagiousStochasticState(
            self.MILD_CONDITION,
            detectable=True,
            contagiousness=self.mild_condition_infection_ratio,
            test_willingness=self.mild_condition_test_willingness,
        )

        need_close_medical_care = ContagiousStochasticState(
            self.NEED_OF_CLOSE_MEDICAL_CARE,
            detectable=True,
            contagiousness=self.need_close_medical_care_infection_ratio,
            test_willingness=self.need_close_medical_care_test_willingness,
        )

        need_icu = ContagiousStochasticState(
            self.NEED_ICU,
            detectable=True,
            contagiousness=self.need_icu_infection_ratio,
            test_willingness=self.need_icu_test_willingness
        )

        improving_health = ContagiousStochasticState(
            self.IMPROVING_HEALTH,
            detectable=True,
            contagiousness=self.improving_health_infection_ratio,
            test_willingness=self.improving_health_test_willingness
        )

        pre_recovered = ContagiousStochasticState(
            self.PRE_RECOVERED,
            detectable=True,
            contagiousness=self.pre_recovered_infection_ratio,
            test_willingness=self.pre_recovered_test_willingness
        )

        deceased = ImmuneTerminalState(
            self.DECEASED, detectable=False, test_willingness=0
        )  # Won't be tested so detectability isn't relevant
        recovered = ImmuneTerminalState(self.RECOVERED, detectable=False,
                                        test_willingness=self.recovered_test_willingness)

        ret = MedicalStateMachine(susceptible, latent)

        latent.add_transfer(
            asymptomatic,
            duration=self.latent_to_asymptomatic_days,
            probability=self.latent_to_asymptomatic_prob
        )
        latent.add_transfer(
            pre_symptomatic,
            duration=self.latent_to_pre_symptomatic_days,
            probability=self.latent_to_pre_symptomatic_prob
        )

        pre_symptomatic.add_transfer(
            mild_condition,
            duration=self.pre_symptomatic_to_mild_days,
            probability=self.pre_symptomatic_to_mild_prob
        )

        mild_condition.add_transfer(
            need_close_medical_care,
            duration=self.mild_to_close_medical_care_days,
            probability=self.mild_to_close_medical_care_prob
        )
        mild_condition.add_transfer(
            need_icu,
            duration=self.mild_to_need_icu_days,
            probability=self.mild_to_need_icu_prob
        )
        mild_condition.add_transfer(
            pre_recovered,
            duration=self.mild_to_pre_recovered_days,
            probability=self.mild_to_pre_recovered_prob
        )

        need_close_medical_care.add_transfer(
            need_icu,
            duration=self.close_medical_care_to_icu_days,
            probability=self.close_medical_care_to_icu_prob
        )
        need_close_medical_care.add_transfer(
            mild_condition,
            duration=self.close_medical_care_to_mild_days,
            probability=self.close_medical_care_to_mild_prob
        )

        need_icu.add_transfer(
            deceased,
            duration=self.need_icu_to_deceased_days,
            probability=self.need_icu_to_deceased_prob
        )
        need_icu.add_transfer(
            improving_health,
            duration=self.need_icu_to_improving_days,
            probability=self.need_icu_to_improving_prob
        )

        improving_health.add_transfer(
            need_icu,
            duration=self.improving_to_need_icu_days,
            probability=self.improving_to_need_icu_prob
        )
        improving_health.add_transfer(
            pre_recovered,
            duration=self.improving_to_pre_recovered_days,
            probability=self.improving_to_pre_recovered_prob
        )
        improving_health.add_transfer(
            mild_condition,
            duration=self.improving_to_mild_condition_days,
            probability=self.improving_to_mild_condition_prob
        )

        pre_recovered.add_transfer(
            recovered,
            duration=self.pre_recovered_to_recovered_days,
            probability=self.pre_recovered_to_recovered_prob
        )

        asymptomatic.add_transfer(
            recovered,
            duration=self.asymptomatic_to_recovered_days,
            probability=self.asymptomatic_to_recovered_prob
        )

        return ret

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__
