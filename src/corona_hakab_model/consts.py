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
from numpy.random import random
from policies_manager import ConditionedPolicy, Policy
from state_machine import StochasticState, TerminalState
from util import dist, rv_discrete, upper_bound


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
    # Tsvika: Currently the distribution is selected based on the number of input parameters.
    # Think we should do something more readable later on.
    # For example: "latent_to_silent_days": {"type":"uniform","lower_bound":1,"upper_bound":3}
    # disease states transition lengths distributions
    latent_to_silent_days: rv_discrete = dist(1, 3)
    silent_to_asymptomatic_days: rv_discrete = dist(0, 3, 10)
    silent_to_symptomatic_days: rv_discrete = dist(0, 3, 10)
    asymptomatic_to_recovered_days: rv_discrete = dist(3, 5, 7)
    symptomatic_to_asymptomatic_days: rv_discrete = dist(7, 10, 14)
    symptomatic_to_hospitalized_days: rv_discrete = dist(0, 1.5, 10)
    hospitalized_to_asymptomatic_days: rv_discrete = dist(18)
    hospitalized_to_icu_days: rv_discrete = dist(5)
    icu_to_deceased_days: rv_discrete = dist(7)
    icu_to_hospitalized_days: rv_discrete = dist(7)
    # average probability for transitions:
    silent_to_asymptomatic_probability: float = 0.2
    symptomatic_to_asymptomatic_probability: float = 0.85
    hospitalized_to_asymptomatic_probability: float = 0.8
    icu_to_hospitalized_probability: float = 0.65
    # infections ratios
    symptomatic_infection_ratio: float = 0.75
    asymptomatic_infection_ratio: float = 0.25
    silent_infection_ratio: float = 0.3
    # base r0 of the disease
    r0: float = 2.4

    # --Detection tests params--
    # the probability that an infected agent is asking to be tested
    susceptible_test_willingness: float = 0.01
    latent_test_willingness: float = 0.01
    silent_test_willingness: float = 0.01
    asymptomatic_test_willingness: float = 0.01
    symptomatic_test_willingness: float = 0.6
    hospitalized_test_willingness: float = 0.9
    icu_test_willingness: float = 1.0
    recovered_test_willingness: float = 0.1
    detection_pool: List[DetectionTest] = [
                                              DetectionSettings(
                                                  name="hospital",
                                                  detection_test=DetectionTest(detection_prob=0.98,
                                                                               false_alarm_prob=0.02,
                                                                               time_until_result=3),
                                                  daily_num_of_tests_schedule={0: 100, 10: 1000, 20: 2000, 50: 5000},
                                                  testing_gap_after_positive_test=10,
                                                  testing_gap_after_negative_test=5,
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
                                                                               false_alarm_prob=0.03,
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

    # --policies params--
    change_policies: bool = False
    # a dictionary of day:([ConnectionTypes], message). on each day, keeps only the given connection types opened
    policies_changes: Dict[int, tuple] = {
        40: ([ConnectionTypes.Family, ConnectionTypes.Other], "closing schools and works"),
        70: ([ConnectionTypes.Family, ConnectionTypes.Other, ConnectionTypes.School], "opening schools"),
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
            "DetectionTest": DetectionTest,
            "ConditionedPolicy": ConditionedPolicy,
            "ConnectionTypes": ConnectionTypes,
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

    @property
    def silent_to_symptomatic_probability(self) -> float:
        return 1 - self.silent_to_asymptomatic_probability

    @property
    def symptomatic_to_hospitalized_probability(self) -> float:
        return 1 - self.symptomatic_to_asymptomatic_probability

    @property
    def hospitalized_to_icu_probability(self) -> float:
        return 1 - self.hospitalized_to_asymptomatic_probability

    @property
    def icu_to_dead_probability(self) -> float:
        return 1 - self.icu_to_hospitalized_probability

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

        susceptible = SusceptibleTerminalState("Susceptible", test_willingness=self.susceptible_test_willingness)
        latent = ImmuneStochasticState("Latent", detectable=False, test_willingness=self.latent_test_willingness)
        silent = ContagiousStochasticState(
            "Silent", contagiousness=self.silent_infection_ratio, test_willingness=self.silent_test_willingness
        )
        symptomatic = ContagiousStochasticState(
            "Symptomatic",
            contagiousness=self.symptomatic_infection_ratio,
            test_willingness=self.symptomatic_test_willingness,
        )
        asymptomatic = ContagiousStochasticState(
            "Asymptomatic",
            contagiousness=self.asymptomatic_infection_ratio,
            test_willingness=self.asymptomatic_test_willingness,
        )

        hospitalized = ImmuneStochasticState(
            "Hospitalized", detectable=True, test_willingness=self.hospitalized_test_willingness
        )
        icu = ImmuneStochasticState("ICU", detectable=True, test_willingness=self.icu_test_willingness)

        deceased = ImmuneTerminalState(
            "Deceased", detectable=False, test_willingness=0
        )  # Won't be tested so detectability isn't relevant
        recovered = ImmuneTerminalState("Recovered", detectable=False, test_willingness=self.recovered_test_willingness)

        ret = MedicalStateMachine(susceptible, latent)

        latent.add_transfer(silent, self.latent_to_silent_days, ...)

        silent.add_transfer(
            asymptomatic, self.silent_to_asymptomatic_days, self.silent_to_asymptomatic_probability,
        )
        silent.add_transfer(symptomatic, self.silent_to_symptomatic_days, ...)

        symptomatic.add_transfer(
            asymptomatic, self.symptomatic_to_asymptomatic_days, self.symptomatic_to_asymptomatic_probability,
        )
        symptomatic.add_transfer(hospitalized, self.symptomatic_to_hospitalized_days, ...)

        hospitalized.add_transfer(icu, self.hospitalized_to_icu_days, self.hospitalized_to_icu_probability)
        hospitalized.add_transfer(asymptomatic, self.hospitalized_to_asymptomatic_days, ...)

        icu.add_transfer(
            hospitalized, self.icu_to_hospitalized_days, self.icu_to_hospitalized_probability,
        )
        icu.add_transfer(deceased, self.icu_to_deceased_days, ...)

        asymptomatic.add_transfer(recovered, self.asymptomatic_to_recovered_days, ...)

        return ret

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__


# TODO can we remove it?
if __name__ == "__main__":
    c = Consts()
    print(c.average_time_in_each_state())
