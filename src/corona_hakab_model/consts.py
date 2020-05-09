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
    MILD_CONDITION_BEGIN: str = "Mild-Condition-Begin"
    MILD_CONDITION_END: str = "Mild-Condition-End"
    PRE_SYMPTOMATIC: str = "Pre-Symptomatic"
    ASYMPTOMATIC_BEGIN: str = "AsymptomaticBegin"
    ASYMPTOMATIC_END: str = "AsymptomaticEnd"
    LATENT_ASYMP: str = "Latent-Asymp"
    LATENT_PRESYMP: str = "Latent-Presymp"
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
    latent_presymp_to_pre_symptomatic_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 3, 10)})

    latent_to_latent_asymp_begin_days: BucketDict[int, Callable] = BucketDict({0: dist(1)})
    latent_to_latent_presymp_begin_days: BucketDict[int, Callable] = BucketDict({0: dist(1)})

    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10],
    # [0.022,0.052,0.082,0.158,0.234,0.158,0.152,0.082,0.04,0.02]))
    latent_asym_to_asymptomatic_begin_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 3, 10)})
    # latent_asym_to_asymptomatic_begin_days: BucketDict[int, Callable] = BucketDict({
    #     0: dist(
    #         [(2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5), (0, 6), (1, 6), (2, 6), (3, 6), (0, 7), (1, 7), (2, 7), (3, 7), (0, 8), (1, 8), (2, 8), (3, 8), (0, 9), (1, 9), (2, 9), (3, 9), (0, 10), (1, 10), (2, 10), (3, 10), (0, 11), (1, 11), (2, 11), (3, 11), (0, 12), (1, 12), (2, 12), (3, 12), (0, 13), (1, 13), (2, 13), (3, 13)]
    #         [0.0065, 0.0065, 0.0310, 0.0310, 0.0310, 0.0455, 0.0455, 0.0455, 0.0455, 0.0503, 0.0503, 0.0503, 0.0503, 0.0425, 0.0425, 0.0425, 0.0425, 0.0308, 0.0308, 0.0308, 0.0308, 0.0205, 0.0205, 0.0205, 0.0205, 0.0130, 0.0130, 0.0130, 0.0130, 0.0083, 0.0083, 0.0083, 0.0083, 0.0050, 0.0050, 0.0050, 0.0050, 0.0030, 0.0030, 0.0030, 0.0030, 0.0047, 0.0047, 0.0047, 0.0047]
    #     )
    # })

    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11],
    # [0.02,0.05,0.08,0.15,0.22,0.15,0.15,0.08,0.05,0.03,0.02]))
    asymptomatic_begin_to_asymptomatic_end_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 3, 5)})
    pre_symptomatic_to_mild_condition_begin_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 3)})
    mild_condition_begin_to_mild_condition_end_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 3, 5)})
    mild_end_to_close_medical_care_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 8)})
    # Actual distribution: rv_discrete(values=([3,4,5,6,7,8,9,10,11,12],
    # [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.01]))
    mild_end_to_need_icu_days: BucketDict[int, Callable] = BucketDict({0: dist(3, 10, 26)})
    # Actual distribution: rv_discrete(values=([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    # [0.012,0.019,0.032,0.046,0.059,0.069,0.076,0.078,0.076,0.072,0.066,0.060,0.053,0.046,0.040,0.035,0.030,0.028,0.026,0.022,0.020,0.015,0.010,0.010]))
    mild_end_to_pre_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 13, 23)})
    # Actual distribution: rv_discrete(values=(
    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
    # [0.001,0.001,0.001,0.001,0.001,0.002,0.004,0.008,0.013,0.022,0.032,0.046,0.06,0.075,0.088,0.097,0.1,0.098,0.088,0.075,0.06,0.046,0.032,0.022,0.013,0.008,0.004,0.002]))
    close_medical_care_to_icu_days: BucketDict[int, Callable] = BucketDict({0: dist(10, 12, 14)})
    close_medical_care_to_mild_end_days: BucketDict[int, Callable] = BucketDict({0: dist(8, 10, 12)})
    need_icu_to_deceased_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 3, 20)})
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    # [0.030,0.102,0.126,0.112,0.090,0.080,0.075,0.070,0.065,0.050,0.040,0.035,0.030,0.025,0.020,
    # 0.015,0.012,0.010,0.008,0.005]))
    need_icu_to_improving_days: BucketDict[int, Callable] = BucketDict({0: dist(1, 5, 25)})
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
    # [0.021,0.041,0.081,0.101,0.101,0.081,0.071,0.066,0.061,0.056,0.046,0.041,0.039,0.033,0.031,0.026,0.021,0.016,0.013,0.013,0.011,0.011,0.009,0.005,0.005]))
    improving_to_need_icu_days: BucketDict[int, Callable] = BucketDict({0: dist(21, 42)})
    improving_to_pre_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(21, 42)})  # TODO: check why so long
    improving_to_mild_condition_end_days: BucketDict[int, Callable] = BucketDict({0: dist(21, 42)})
    pre_recovered_to_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(14, 28)})
    # Actual distribution: rv_discrete(values=([14, 28], [0.8, 0.2]))
    asymptomatic_end_to_recovered_days: BucketDict[int, Callable] = BucketDict({0: dist(10, 18, 35)})
    # Actual distribution: rv_discrete(values=(
    # [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    # [0.013,0.016,0.025,0.035,0.045,0.053,0.061,0.065,0.069,0.069,0.065,0.063,0.058,0.053,0.056,0.041,0.040,0.033,
    # 0.030,0.025,0.020,0.015,0.015,0.015,0.010,0.010]))
    # state machine transfer probabilities
    # probability of '...' equals (1 - all other transfers)
    # it should always come last after all other transition probabilities were defined
    latent_to_latent_asymp_begin_prob: TransitionProbType = BucketDict({0: 0.3})
    asymptomatic_begin_to_asymptomatic_end_prob:  TransitionProbType = BucketDict({0: ...})
    latent_to_latent_presymp_prob:  TransitionProbType = BucketDict({0: ...})
    latent_presymp_to_pre_symptomatic_prob:  TransitionProbType = BucketDict({0: ...})
    latent_asym_to_asymptomatic_begin_prob:  TransitionProbType = BucketDict({0: ...})
    pre_symptomatic_to_mild_condition_begin_prob:  TransitionProbType = BucketDict({0: ...})
    mild_condition_begin_to_mild_condition_end_prob:  TransitionProbType = BucketDict({0: ...})
    mild_end_to_close_medical_care_prob:  TransitionProbType = BucketDict({0: 0.2375})
    mild_end_to_need_icu_prob:  TransitionProbType = BucketDict({0: 0.0324})
    mild_end_to_pre_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    close_medical_care_to_icu_prob:  TransitionProbType = BucketDict({0: 0.26})
    close_medical_care_to_mild_end_prob:  TransitionProbType = BucketDict({0: ...})
    need_icu_to_deceased_prob:  TransitionProbType = BucketDict({0: 0.3})
    need_icu_to_improving_prob:  TransitionProbType = BucketDict({0: ...})
    improving_to_need_icu_prob:  TransitionProbType = BucketDict({0: 0})
    improving_to_pre_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    improving_to_mild_condition_end_prob:  TransitionProbType = BucketDict({0: 0})
    pre_recovered_to_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    asymptomatic_end_to_recovered_prob:  TransitionProbType = BucketDict({0: ...})
    # infections ratios, See bucket dict for more info on how to use.
    pre_symptomatic_infection_ratio: BucketDict[int, int] = BucketDict({0: [0.14, 0.86, 1]})  # if x greater than biggest key, x is biggest key
    asymptomatic_begin_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0.14, 0.86, 1, 0.82, 0.59, 0.41, 0.27, 0.18, 0.14, 0.09, 0.05]})
    mild_condition_begin_infection_ratio: BucketDict[int, int] = BucketDict({0: [0.82, 0.59, 0.41, 0.27, 0.18, 0.14, 0.09, 0.05]})
    latent_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    mild_condition_end_infection_ratio: BucketDict[int, int] = BucketDict({0: [0]})
    latent_presymp_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    latent_asymp_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
    asymptomatic_end_infection_ratio:  BucketDict[int, int] = BucketDict({0: [0]})
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
    asymptomatic_begin_test_willingness: float = 0.01
    asymptomatic_end_test_willingness: float = 0.01
    pre_symptomatic_test_willingness: float = 0.01
    mild_condition_begin_test_willingness: float = 0.6
    mild_condition_end_test_willingness: float = 0.6
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
                MILD_CONDITION_BEGIN: .98,
                MILD_CONDITION_END: .98,
                PRE_SYMPTOMATIC: .98,
                ASYMPTOMATIC_BEGIN: .98,
                ASYMPTOMATIC_END: .98,
                LATENT_ASYMP: .98,
                LATENT_PRESYMP: .98
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
                MILD_CONDITION_BEGIN: .92,
                MILD_CONDITION_END: .92,
                PRE_SYMPTOMATIC: .92,
                ASYMPTOMATIC_BEGIN: .92,
                ASYMPTOMATIC_END: .92,
                LATENT_ASYMP: .92,
                LATENT_PRESYMP: .92
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
        latent_presymp = ContagiousStochasticState(
            self.LATENT_PRESYMP,
            detectable=False,
            contagiousness=self.latent_presymp_infection_ratio,
            test_willingness=self.latent_test_willingness
        )
        latent_asymp = ContagiousStochasticState(
            self.LATENT_ASYMP,
            detectable=False,
            contagiousness=self.latent_asymp_infection_ratio,
            test_willingness=self.latent_test_willingness
        )
        asymptomatic_begin = ContagiousStochasticState(
            self.ASYMPTOMATIC_BEGIN,
            detectable=True,
            contagiousness=self.asymptomatic_begin_infection_ratio,
            test_willingness=self.asymptomatic_begin_test_willingness
        )
        asymptomatic_end = ContagiousStochasticState(
            self.ASYMPTOMATIC_END,
            detectable=True,
            contagiousness=self.asymptomatic_end_infection_ratio,
            test_willingness=self.asymptomatic_end_test_willingness
        )
        pre_symptomatic = ContagiousStochasticState(
            self.PRE_SYMPTOMATIC,
            detectable=True,
            contagiousness=self.pre_symptomatic_infection_ratio,
            test_willingness=self.pre_symptomatic_test_willingness,
        )
        mild_condition_begin = ContagiousStochasticState(
            self.MILD_CONDITION_BEGIN,
            detectable=True,
            contagiousness=self.mild_condition_begin_infection_ratio,
            test_willingness=self.mild_condition_begin_test_willingness,
        )
        mild_condition_end = ContagiousStochasticState(
            self.MILD_CONDITION_END,
            detectable=True,
            contagiousness=self.mild_condition_end_infection_ratio,
            test_willingness=self.mild_condition_end_test_willingness,
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
            latent_asymp,
            duration=self.latent_to_latent_asymp_begin_days,
            probability=self.latent_to_latent_asymp_begin_prob
        )
        latent.add_transfer(
            latent_presymp,
            duration=self.latent_to_latent_presymp_begin_days,
            probability=self.latent_to_latent_presymp_prob
        )

        latent_presymp.add_transfer(
            pre_symptomatic,
            duration=self.latent_presymp_to_pre_symptomatic_days,
            probability=self.latent_presymp_to_pre_symptomatic_prob
        )

        latent_asymp.add_transfer(
            asymptomatic_begin,
            duration=self.latent_asym_to_asymptomatic_begin_days,
            probability=self.latent_asym_to_asymptomatic_begin_prob
        )

        asymptomatic_begin.add_transfer(
            asymptomatic_end,
            duration=self.asymptomatic_begin_to_asymptomatic_end_days,
            probability=self.asymptomatic_begin_to_asymptomatic_end_prob
        )

        pre_symptomatic.add_transfer(
            mild_condition_begin,
            duration=self.pre_symptomatic_to_mild_condition_begin_days,
            probability=self.pre_symptomatic_to_mild_condition_begin_prob
        )

        mild_condition_begin.add_transfer(
            mild_condition_end,
            duration=self.mild_condition_begin_to_mild_condition_end_days,
            probability=self.mild_condition_begin_to_mild_condition_end_prob
        )

        mild_condition_end.add_transfer(
            need_close_medical_care,
            duration=self.mild_end_to_close_medical_care_days,
            probability=self.mild_end_to_close_medical_care_prob
        )
        mild_condition_end.add_transfer(
            need_icu,
            duration=self.mild_end_to_need_icu_days,
            probability=self.mild_end_to_need_icu_prob
        )
        mild_condition_end.add_transfer(
            pre_recovered,
            duration=self.mild_end_to_pre_recovered_days,
            probability=self.mild_end_to_pre_recovered_prob
        )

        need_close_medical_care.add_transfer(
            need_icu,
            duration=self.close_medical_care_to_icu_days,
            probability=self.close_medical_care_to_icu_prob
        )
        need_close_medical_care.add_transfer(
            mild_condition_end,
            duration=self.close_medical_care_to_mild_end_days,
            probability=self.close_medical_care_to_mild_end_prob
        )

        need_icu.add_transfer(
            deceased,
            self.need_icu_to_deceased_days,
            probability=self.need_icu_to_deceased_prob
        )
        need_icu.add_transfer(
            improving_health,
            self.need_icu_to_improving_days,
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
            mild_condition_end,
            duration=self.improving_to_mild_condition_end_days,
            probability=self.improving_to_mild_condition_end_prob
        )

        pre_recovered.add_transfer(
            recovered,
            duration=self.pre_recovered_to_recovered_days,
            probability=self.pre_recovered_to_recovered_prob
        )

        asymptomatic_end.add_transfer(
            recovered,
            duration=self.asymptomatic_end_to_recovered_days,
            probability=self.asymptomatic_end_to_recovered_prob
        )

        return ret

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__
