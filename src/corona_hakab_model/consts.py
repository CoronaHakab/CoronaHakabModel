from functools import lru_cache
from itertools import count
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.random import random

from detection_model.detection_testing_types import DetectionSettings, DetectionPriority
from detection_model.healthcare import DetectionTest
from generation.connection_types import ConnectionTypes
from medical_state import ContagiousState, ImmuneState, MedicalState, SusceptibleState
from medical_state_machine import MedicalStateMachine
from policies_manager import ConditionedPolicy, Policy
from state_machine import StochasticState, TerminalState
from util import dist, rv_discrete, upper_bound, BucketDict

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
    # attributes and default values:

    total_steps: int = 350
    initial_infected_count: int = 20
    export_infected_agents_interval: int = 50

    # Size of population to estimate expected time for each state
    population_size_for_state_machine_analysis: int = 25_000

    # Tsvika: Currently the distribution is selected based on the number of input parameters.
    # Think we should do something more readable later on.
    # For example: "latent_to_silent_days": {"type":"uniform","lower_bound":1,"upper_bound":3}
    # disease states transition lengths distributions
    latent_to_pre_symptomatic_days: rv_discrete = dist(1, 5, 10)
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10],
    # [0.022,0.052,0.082,0.158,0.234,0.158,0.152,0.082,0.04,0.02]))
    latent_to_asymptomatic_days: rv_discrete = dist(1, 5, 11)
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11],
    # [0.02,0.05,0.08,0.15,0.22,0.15,0.15,0.08,0.05,0.03,0.02]))
    pre_symptomatic_to_mild_condition_days: rv_discrete = dist(1, 3)
    mild_to_close_medical_care_days: rv_discrete = dist(3, 11)
    # Actual distribution: rv_discrete(values=([3,4,5,6,7,8,9,10,11,12],
    # [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.01]))
    mild_to_need_icu_days: rv_discrete = dist(6, 13, 29)
    # Actual distribution: rv_discrete(values=([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    # [0.012,0.019,0.032,0.046,0.059,0.069,0.076,0.078,0.076,0.072,0.066,0.060,0.053,0.046,0.040,0.035,0.030,0.028,0.026,0.022,0.020,0.015,0.010,0.010]))
    mild_to_pre_recovered_days: rv_discrete = dist(1, 17, 28)
    # Actual distribution: rv_discrete(values=(
    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
    # [0.001,0.001,0.001,0.001,0.001,0.002,0.004,0.008,0.013,0.022,0.032,0.046,0.06,0.075,0.088,0.097,0.1,0.098,0.088,0.075,0.06,0.046,0.032,0.022,0.013,0.008,0.004,0.002]))
    close_medical_care_to_icu_days: rv_discrete = dist(10, 12, 14)
    close_medical_care_to_mild_days: rv_discrete = dist(8, 10, 12)
    need_icu_to_deceased_days: rv_discrete = dist(1, 3, 20)
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    # [0.030,0.102,0.126,0.112,0.090,0.080,0.075,0.070,0.065,0.050,0.040,0.035,0.030,0.025,0.020,
    # 0.015,0.012,0.010,0.008,0.005]))
    need_icu_to_improving_days: rv_discrete = dist(1, 5, 25)
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
    # [0.021,0.041,0.081,0.101,0.101,0.081,0.071,0.066,0.061,0.056,0.046,0.041,0.039,0.033,0.031,0.026,0.021,0.016,0.013,0.013,0.011,0.011,0.009,0.005,0.005]))
    improving_to_need_icu_days: rv_discrete = dist(21, 42)
    improving_to_pre_recovered_days: rv_discrete = dist(21, 42)
    improving_to_mild_condition_days: rv_discrete = dist(21, 42)
    pre_recovered_to_recovered_days: rv_discrete = dist(14, 28)
    # Actual distribution: rv_discrete(values=([14, 28], [0.8, 0.2]))
    asymptomatic_to_recovered_days: rv_discrete = dist(10, 18, 35)
    # Actual distribution: rv_discrete(values=(
    # [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    # [0.013,0.016,0.025,0.035,0.045,0.053,0.061,0.065,0.069,0.069,0.065,0.063,0.058,0.053,0.056,0.041,0.040,0.033,
    # 0.030,0.025,0.020,0.015,0.015,0.015,0.010,0.010]))
    # infections ratios, See bucket dict for more info on how to use.
    pre_symptomatic_infection_ratio: BucketDict = BucketDict({10: 0.75, 20: 0.75})  # x <= 10 then key is 10,
    mild_condition_infection_ratio: BucketDict = BucketDict({10: 0.40})  # x<=20 then key is 20,
    latent_infection_ratio: BucketDict = BucketDict({0: 0})   # if x greater than biggest key, x is biggest key
    latent_presymp_infection_ratio: BucketDict = BucketDict({0: 0})
    latent_asymp_infection_ratio: BucketDict = BucketDict({0: 0})
    asymptomatic_infection_ratio: BucketDict = BucketDict({0: 0})
    need_close_medical_care_infection_ratio: BucketDict = BucketDict({0: 0})
    need_icu_infection_ratio: BucketDict = BucketDict({0: 0})
    improving_health_infection_ratio: BucketDict = BucketDict({0: 0})
    pre_recovered_infection_ratio: BucketDict = BucketDict({0: 0})
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
            detection_test=DetectionTest(detection_prob=0.98,
                                         false_alarm_prob=0.,
                                         time_until_result=3),
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
            detection_test=DetectionTest(detection_prob=0.92,
                                         false_alarm_prob=0.,
                                         time_until_result=5),
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
    should_isolate_positive_detected = False
    isolate_after_num_day = 1  # will be in isolation the next day
    p_will_obey_isolation = 1.0  # 100% will obey the isolation
    isolation_factor = 0.0  # reduce 100%, meaning mult by 0

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
    partial_opening_active: bool = True
    # each connection type gets a list of conditioned policies.
    # each conditioned policy actives a specific policy when a condition is satisfied.
    # each policy changes the multiplication factor of a specific circle.
    # each policy is activated only if a list of terms is fulfilled.
    connection_type_to_conditioned_policy: Dict[ConnectionTypes, List[ConditionedPolicy]] = {
        ConnectionTypes.School: [
            ConditionedPolicy(
                activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
                policy=Policy(0, [lambda circle: random() > 0]),
                message="closing all schools",
            ),
            ConditionedPolicy(
                activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
                policy=Policy(1, [lambda circle: random() > 1]),
                active=True,
                message="opening all schools",
            ),
        ],
        ConnectionTypes.Kindergarten: [
            ConditionedPolicy(
                activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
                policy=Policy(0, [lambda circle: random() > 0]),
                message="closing all kindergartens",
            ),
            ConditionedPolicy(
                activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
                policy=Policy(1, [lambda circle: random() > 1]),
                active=True,
                message="opening all kindergartens",
            ),
        ],
        ConnectionTypes.Work: [
            ConditionedPolicy(
                activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
                policy=Policy(0, [lambda circle: random() > 0]),
                message="closing all workplaces",
            ),
            ConditionedPolicy(
                activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
                policy=Policy(0, [lambda circle: random() > 1]),
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
            "rv_discrete": rv_discrete,
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

    @lru_cache(None)
    def average_time_in_each_state(self) -> Dict[MedicalState, int]:
        """
        calculate the average time an infected agent spends in any of the states.
        uses markov chain to do the calculations
        note that it doesnt work well for terminal states
        :return: dict of states: int, representing the average time an agent would be in a given state
        """
        TOL = 1e-6
        m = self.medical_state_machine()
        M, terminal_states, transfer_states, entry_columns = m.markovian
        z = len(M)

        p = entry_columns[m.default_state_upon_infection]
        terminal_mask = np.zeros(z, bool)
        terminal_mask[list(terminal_states.values())] = True

        states_duration: Dict[MedicalState, int] = Dict.fromkeys(m.states, 0)
        states_duration[m.default_state_upon_infection] = 1

        index_to_state: Dict[int, MedicalState] = {}
        for state, index in terminal_states.items():
            index_to_state[index] = state
        for state, dict in transfer_states.items():
            first_index = dict[0]
            last_index = dict[max(dict.keys())] + upper_bound(state.durations[-1])
            for index in range(first_index, last_index):
                index_to_state[index] = state

        prev_v = 0.0
        for time in count(1):
            p = M @ p
            v = np.sum(p, where=terminal_mask)
            d = v - prev_v
            prev_v = v

            for i, prob in enumerate(p):
                states_duration[index_to_state[i]] += prob

            # run at least as many times as the node number to ensure we reached all terminal nodes
            if time > z and d < TOL:
                break
        return states_duration

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

        susceptible = SusceptibleTerminalState("Susceptible", test_willingness=self.susceptible_test_willingness)
        latent = ContagiousStochasticState(
            "Latent",
            detectable=False,
            contagiousness=self.latent_infection_ratio,
            test_willingness=self.latent_test_willingness)
        latent_presymp = ContagiousStochasticState(
            "Latent-Presymp",
            detectable=False,
            contagiousness=self.latent_presymp_infection_ratio,
            test_willingness=self.latent_test_willingness
        )
        latent_asymp = ContagiousStochasticState(
            "Latent-Asymp",
            detectable=False,
            contagiousness=self.latent_asymp_infection_ratio,
            test_willingness=self.latent_test_willingness
        )
        asymptomatic = ContagiousStochasticState(
            "Asymptomatic",
            detectable=True,
            contagiousness=self.asymptomatic_infection_ratio,
            test_willingness=self.asymptomatic_test_willingness
        )
        pre_symptomatic = ContagiousStochasticState(
            "Pre-Symptomatic",
            detectable=True,
            contagiousness=self.pre_symptomatic_infection_ratio,
            test_willingness=self.pre_symptomatic_test_willingness,
        )
        mild_condition = ContagiousStochasticState(
            "Mild-Condition",
            detectable=True,
            contagiousness=self.mild_condition_infection_ratio,
            test_willingness=self.mild_condition_test_willingness,
        )
        need_close_medical_care = ContagiousStochasticState(
            "NeedOfCloseMedicalCare",
            detectable=True,
            contagiousness=self.need_close_medical_care_infection_ratio,
            test_willingness=self.need_close_medical_care_test_willingness,
        )

        need_icu = ContagiousStochasticState(
            "NeedICU",
            detectable=True,
            contagiousness=self.need_icu_infection_ratio,
            test_willingness=self.need_icu_test_willingness
        )

        improving_health = ContagiousStochasticState(
            "ImprovingHealth",
            detectable=True,
            contagiousness=self.improving_health_infection_ratio,
            test_willingness=self.improving_health_test_willingness
        )

        pre_recovered = ContagiousStochasticState(
            "PreRecovered",
            detectable=True,
            contagiousness=self.pre_recovered_infection_ratio,
            test_willingness=self.pre_recovered_test_willingness
        )

        deceased = ImmuneTerminalState(
            "Deceased", detectable=False, test_willingness=0
        )  # Won't be tested so detectability isn't relevant
        recovered = ImmuneTerminalState("Recovered", detectable=False, test_willingness=self.recovered_test_willingness)

        ret = MedicalStateMachine(susceptible, latent)

        latent.add_transfer(
            latent_asymp,
            duration=dist(1),
            probability=0.3
        )
        latent.add_transfer(
            latent_presymp,
            duration=dist(1),
            probability=...
        )

        latent_presymp.add_transfer(
            pre_symptomatic,
            duration=self.latent_to_pre_symptomatic_days,
            probability=...
        )

        latent_asymp.add_transfer(
            asymptomatic,
            duration=self.latent_to_asymptomatic_days,
            probability=...
        )

        pre_symptomatic.add_transfer(
            mild_condition,
            duration=self.pre_symptomatic_to_mild_condition_days,
            probability=...
        )

        mild_condition.add_transfer(
            need_close_medical_care,
            duration=self.mild_to_close_medical_care_days,
            probability=0.2375,
        )
        mild_condition.add_transfer(
            need_icu,
            duration=self.mild_to_need_icu_days,
            probability=0.0324
        )
        mild_condition.add_transfer(
            pre_recovered,
            duration=self.mild_to_pre_recovered_days,
            probability=...
        )

        need_close_medical_care.add_transfer(
            need_icu,
            duration=self.close_medical_care_to_icu_days,
            probability=0.26
        )
        need_close_medical_care.add_transfer(
            mild_condition,
            duration=self.close_medical_care_to_mild_days,
            probability=...
        )

        need_icu.add_transfer(
            deceased,
            self.need_icu_to_deceased_days,
            probability=0.0227
        )
        need_icu.add_transfer(
            improving_health,
            self.need_icu_to_improving_days,
            probability=...
        )

        improving_health.add_transfer(
            need_icu,
            duration=self.improving_to_need_icu_days,
            probability=0.22
        )
        improving_health.add_transfer(
            pre_recovered,
            duration=self.improving_to_pre_recovered_days,
            probability=0.39
        )
        improving_health.add_transfer(
            mild_condition,
            duration=self.improving_to_mild_condition_days,
            probability=...
        )

        pre_recovered.add_transfer(
            recovered,
            duration=self.pre_recovered_to_recovered_days,
            probability=...
        )

        asymptomatic.add_transfer(
            recovered,
            duration=self.asymptomatic_to_recovered_days,
            probability=...
        )

        return ret

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__


# TODO can we remove it?
if __name__ == "__main__":
    c = Consts()
    for state, time in c.average_time_in_each_state().items():
        print(f"For state {state.name} we have expected {time} days")
