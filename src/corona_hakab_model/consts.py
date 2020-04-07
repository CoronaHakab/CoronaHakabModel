from dataclasses import dataclass
from itertools import count
from typing import Dict

import numpy as np
from generation.connection_types import ConnectionTypes
from healthcare import DetectionTest
from medical_state import ContagiousState, ImmuneState, MedicalState, SusceptibleState
from medical_state_machine import MedicalStateMachine
from numpy.random import random
from policies_manager import ConditionedPolicy, Policy
from state_machine import StochasticState, TerminalState
from sub_matrices import CircularConnectionsMatrix, ClusteredConnectionsMatrix, NonCircularConnectionMatrix
from util import dist, rv_discrete, upper_bound

"""
Usage:
1. Create a default consts object - consts = Consts()
2. Load a parameters file - consts = Consts.from_file(path)
"""


@dataclass
class Consts:
    __slots__ = (
        "total_steps",
        "initial_infected_count",
        "latent_to_silent_days",
        "silent_to_asymptomatic_days",
        "silent_to_symptomatic_days",
        "asymptomatic_to_recovered_days",
        "symptomatic_to_asymptomatic_days",
        "symptomatic_to_hospitalized_days",
        "hospitalized_to_asymptomatic_days",
        "hospitalized_to_icu_days",
        "icu_to_deceased_days",
        "icu_to_hospitalized_days",

        # average probability for transitions:
        "silent_to_asymptomatic_probability",
        "symptomatic_to_asymptomatic_probability",
        "hospitalized_to_asymptomatic_probability",
        "icu_to_hospitalized_probability",

        # the probability that an infected agent is asking to be tested
        "symptomatic_test_willingness",
        "asymptomatic_test_willingness",
        "silent_test_willingness",
        # probability of an infected symptomatic agent infecting others
        "symptomatic_infection_ratio",
        # probability of an infected asymptomatic agent infecting others
        "asymptomatic_infection_ratio",
        # probability of an infected silent agent infecting others
        "silent_infection_ratio",

        # base r0 of the disease
        "r0",

        # tests data
        "detection_test",
        "daily_num_of_tests",
        "testing_gap_after_positive_test",
        "testing_gap_after_negative_test",
        "testing_policy",

        # policies data
        # a dictionary of type day:List[ConnectionTypes]. on each day, keeps only the given connection types opened
        "policies_changes",
        "change_policies",
        # policies acting on a specific connection type, when a term is satisfied
        "partial_opening_active",
        "connection_type_to_conditioned_policy"
    )

    def __init__(self):
        self.total_steps = 350
        self.initial_infected_count = 20
        # Tsvika: Currently the distribution is selected based on the number of input parameters.
        # Think we should do something more readable later on.
        # For example: latent_to_silent_days = {"type":"uniform","lower_bound":1"upper_bound":3}
        self.latent_to_silent_days = dist(1, 3)
        self.silent_to_asymptomatic_days = dist(0, 3, 10)
        self.silent_to_symptomatic_days = dist(0, 3, 10)
        self.asymptomatic_to_recovered_days = dist(3, 5, 7)
        self.symptomatic_to_asymptomatic_days = dist(7, 10, 14)
        self.symptomatic_to_hospitalized_days = dist(0, 1.5, 10)  # todo range not specified in sources
        self.hospitalized_to_asymptomatic_days = dist(18)
        self.hospitalized_to_icu_days = dist(5)  # todo probably has a range
        self.icu_to_deceased_days = dist(7)  # todo probably has a range
        # todo maybe the program should juts print a question mark,  we'll see how the researchers like that!
        self.icu_to_hospitalized_days = dist(7)
        # average probability for transitions:
        self.silent_to_asymptomatic_probability = 0.2
        self.symptomatic_to_asymptomatic_probability = 0.85
        self.hospitalized_to_asymptomatic_probability = 0.8
        self.        icu_to_hospitalized_probability = 0.65
        # the probability that an infected agent is asking to be tested
        self.symptomatic_test_willingness = 0.6
        self.asymptomatic_test_willingness = 0.05
        self.silent_test_willingness = 0.01
        # probability of an infected symptomatic agent infecting others
        self.symptomatic_infection_ratio = 0.75
        # probability of an infected asymptomatic agent infecting others
        self.asymptomatic_infection_ratio = 0.25
        # probability of an infected silent agent infecting others
        self.silent_infection_ratio = 0.3  # todo i made this up need to get the real number
        # base r0 of the disease
        self.r0 = 2.4
        # tests data
        self.detection_test = DetectionTest(detection_prob=0.98, false_alarm_prob=0.02, time_until_result=3)
        self.daily_num_of_tests = 3000
        self.testing_gap_after_positive_test = 4
        self.testing_gap_after_negative_test = 1
        self.testing_policy = (
            lambda agent: agent.medical_state.name == "Recovered",
            lambda agent: agent.medical_state.name == "Symptomatic",
        )  # TODO: Define better API
        # policies data
        # a dictionary of type day:List[ConnectionTypes]. on each day, keeps only the given connection types opened
        self.policies_changes = {
            40: ([ConnectionTypes.Family, ConnectionTypes.Other], "closing schools and works"),
            70: ([ConnectionTypes.Family, ConnectionTypes.Other, ConnectionTypes.School], "opening schools"),
            100: (ConnectionTypes, "opening works"),
        }
        self.change_policies = False
        # policies acting on a specific connection type, when a term is satisfied
        self.partial_opening_active = True
        # each connection type gets a list of conditioned policies.
        # each conditioned policy actives a specific policy when a condition is satisfied.
        # each policy changes the multiplication factor of a specific circle.
        # each policy is activated only if a list of terms is fulfilled.
        self.connection_type_to_conditioned_policy = {
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

    @staticmethod
    def from_file(param_path):
        """
        Load parameters from file and return Consts object with those values.

        No need to sanitize the eval'd data as we disabled __builtins__ and only passed specific functions
        Documentation about what is allowed and not allowed can be found at the top of this page.
        """
        with open(param_path, "rt") as read_file:
            data = read_file.read()

        parameters = eval(data, {"__builtins__": None, "dist": dist, "rv_discrete": rv_discrete})
        Consts.sanitize_parameters(parameters)

        return Consts(**parameters)

    @staticmethod
    def sanitize_parameters(parameters):
        consts = Consts(**parameters)
        try:
            hash(consts)
        except TypeError as e:
            raise TypeError("Unhashable value in parameters") from e

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

    def medical_state_machine(self) -> MedicalStateMachine:
        class SusceptibleTerminalState(SusceptibleState, TerminalState):
            pass

        class ImmuneStochasticState(ImmuneState, StochasticState):
            pass

        class ContagiousStochasticState(ContagiousState, StochasticState):
            pass

        class ImmuneTerminalState(ImmuneState, TerminalState):
            pass

        susceptible = SusceptibleTerminalState("Susceptible")
        latent = ImmuneStochasticState("Latent", detectable=False)
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

        hospitalized = ImmuneStochasticState("Hospitalized", detectable=True)
        icu = ImmuneStochasticState("ICU", detectable=True)

        deceased = ImmuneTerminalState("Deceased", detectable=False)  # Won't be tested so detectability isn't relevant
        recovered = ImmuneTerminalState("Recovered", detectable=False)

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

    @property
    # todo this should be a consts
    def circular_matrices(self):
        return [
            CircularConnectionsMatrix("home", None, self.family_size_distribution, self.family_strength),
        ]

    @property
    # todo this should be a consts
    def non_circular_matrices(self):
        return [
            NonCircularConnectionMatrix("strangers", None, self.strangers_scale_factor, self.stranger_strength),
        ]

    @property
    def clustered_matrices(self):
        return [
            ClusteredConnectionsMatrix("work", None, self.work_scale_factor, self.work_strength),
            ClusteredConnectionsMatrix("school", None, self.school_scale_factor, self.school_strength),
        ]


if __name__ == "__main__":
    c = Consts()
    print(c.average_time_in_each_state())
