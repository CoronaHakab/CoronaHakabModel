import json
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import count
from typing import Dict, List, NamedTuple

import numpy as np

from subconsts.detection_test import DetectionTestConsts
from subconsts.infection_ratios import InfectionRatios
from subconsts.medical_state_transition import MedicalStateTransition

from generation.connection_types import ConnectionTypes
from medical_state import ContagiousState, ImmuneState, MedicalState, SusceptibleState
from medical_state_machine import MedicalStateMachine
from numpy.random import random
from policies_manager import ConditionedPolicy, Policy
from state_machine import StochasticState, TerminalState
from util import upper_bound
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

    infection_ratios = InfectionRatios()

    # base r0 of the disease
    r0: float = 2.4

    medical_state_transition = MedicalStateTransition()

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
    partial_opening_active: bool = False

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
        infection_ratios = kwargs.pop('infection_ratios')
        medical_state_transition = kwargs.pop('medical_state_transition')
        detection_test_consts = kwargs.pop('detection_test_consts')
        policy_changes = kwargs.pop('policy_changes')
        self = cls(**kwargs)
        self.infection_ratios = InfectionRatios(**infection_ratios)
        self.medical_state_transition = cls.medical_state_transition.json_dict_to_instance(**medical_state_transition)
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

        test_willingness_consts = self.detection_test_consts.test_willingness
        susceptible = SusceptibleTerminalState("Susceptible", test_willingness=test_willingness_consts.susceptible)
        latent = ImmuneStochasticState("Latent", detectable=False, test_willingness=test_willingness_consts.latent)
        silent = ContagiousStochasticState(
            "Silent", contagiousness=self.infection_ratios.silent, test_willingness=test_willingness_consts.silent
        )
        symptomatic = ContagiousStochasticState(
            "Symptomatic",
            contagiousness=self.infection_ratios.symptomatic,
            test_willingness=test_willingness_consts.symptomatic,
        )
        asymptomatic = ContagiousStochasticState(
            "Asymptomatic",
            contagiousness=self.infection_ratios.asymptomatic,
            test_willingness=test_willingness_consts.asymptomatic,
        )

        hospitalized = ImmuneStochasticState(
            "Hospitalized", detectable=True, test_willingness=test_willingness_consts.hospitalized
        )
        icu = ImmuneStochasticState("ICU", detectable=True, test_willingness=test_willingness_consts.icu)

        deceased = ImmuneTerminalState(
            "Deceased", detectable=False, test_willingness=0
        )  # Won't be tested so detectability isn't relevant
        recovered = ImmuneTerminalState("Recovered", detectable=False,
                                        test_willingness=test_willingness_consts.recovered)

        ret = MedicalStateMachine(susceptible, latent)

        day_transition_distributions: MedicalStateTransition.DayDistributions = \
            self.medical_state_transition.day_distributions
        transition_probabilities: MedicalStateTransition.TransitionProbabilities = \
            self.medical_state_transition.transition_probabilities

        latent.add_transfer(silent, day_transition_distributions.latent_to_silent, ...)

        silent.add_transfer(
            asymptomatic, day_transition_distributions.silent_to_asymptomatic,
            transition_probabilities.silent_to_asymptomatic,
        )
        silent.add_transfer(symptomatic, day_transition_distributions.silent_to_symptomatic, ...)

        symptomatic.add_transfer(
            asymptomatic, day_transition_distributions.symptomatic_to_asymptomatic,
            transition_probabilities.symptomatic_to_asymptomatic,
        )
        symptomatic.add_transfer(hospitalized, day_transition_distributions.symptomatic_to_hospitalized, ...)

        hospitalized.add_transfer(icu, day_transition_distributions.hospitalized_to_icu,
                                  transition_probabilities.hospitalized_to_icu)
        hospitalized.add_transfer(asymptomatic, day_transition_distributions.hospitalized_to_asymptomatic, ...)

        icu.add_transfer(
            hospitalized, day_transition_distributions.icu_to_hospitalized,
            transition_probabilities.icu_to_hospitalized,
        )
        icu.add_transfer(deceased, day_transition_distributions.icu_to_deceased, ...)

        asymptomatic.add_transfer(recovered, day_transition_distributions.asymptomatic_to_recovered, ...)

        return ret

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__


# TODO can we remove it?
if __name__ == "__main__":
    c = Consts()
    print(c.average_time_in_each_state())
