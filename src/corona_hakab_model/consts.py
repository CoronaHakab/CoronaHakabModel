from functools import lru_cache
from itertools import count
from typing import NamedTuple, Dict, List

import numpy as np
from medical_state import ImmuneState, SusceptibleState, ContagiousState, MedicalState
from medical_state_machine import MedicalStateMachine
from scipy.stats import rv_discrete
from state_machine import StochasticState, TerminalState
from util import dist, upper_bound


class Consts(NamedTuple):
    # simulation parameters
    population_size = 10_000
    total_steps = 350
    initial_infected_count = 20

    # corona stats
    # todo replace with distribution
    # average state mechine transmitions times:
    latent_to_silent_days: rv_discrete = dist(1, 3)
    silent_to_asymptomatic_days: rv_discrete = dist(0, 3, 10)
    silent_to_symptomatic_days: rv_discrete = dist(0, 3, 10)
    asymptomatic_to_recovered_days: rv_discrete = dist(3, 5, 7)
    symptomatic_to_asymptomatic_days: rv_discrete = dist(7, 10, 14)
    symptomatic_to_hospitalized_days: rv_discrete = dist(
        0, 1.5, 10
    )  # todo range not specified in sources
    hospitalized_to_asymptomatic_days: rv_discrete = dist(18)
    hospitalized_to_icu_days: rv_discrete = dist(5)  # todo probably has a range
    icu_to_deceased_days: rv_discrete = dist(7)  # todo probably has a range
    icu_to_hospitalized_days: rv_discrete = dist(
        7
    )  # todo maybe the program should juts print a question mark,  we'll see how the researchers like that!
    detection_rate = 0.2

    def average_time_in_each_state(self):
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

        p = entry_columns[m.state_upon_infection]
        terminal_mask = np.zeros(z, bool)
        terminal_mask[list(terminal_states.values())] = True

        states_duration: Dict[MedicalState: int] = Dict.fromkeys(m.states, 0)
        states_duration[m.state_upon_infection] = 1

        index_to_state: Dict[int: MedicalState] = {}
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

    # average probability for transmitions:
    silent_to_asymptomatic_probability = 0.2

    @property
    def silent_to_symptomatic_probability(self):
        return 1 - self.silent_to_asymptomatic_probability

    symptomatic_to_asymptomatic_probability = 0.85

    @property
    def symptomatic_to_hospitalized_probability(self):
        return 1 - self.symptomatic_to_asymptomatic_probability

    hospitalized_to_asymptomatic_probability = 0.8

    @property
    def hospitalized_to_icu_probability(self):
        return 1 - self.hospitalized_to_asymptomatic_probability

    icu_to_hospitalized_probability = 0.65

    @property
    def icu_to_dead_probability(self):
        return 1 - self.icu_to_hospitalized_probability

    # probability of an infected symptomatic agent infecting others
    symptomatic_infection_ratio: float = 0.75
    # probability of an infected asymptomatic agent infecting others
    asymptomatic_infection_ratio: float = 0.25
    # probability of an infected silent agent infecting others
    silent_infection_ratio: float = 0.3  # todo i made this up, need to get the real number
    # base r0 of the disease
    r0: float = 2.4

    # isolation policy
    # todo why does this exist? doesn't the policy set this? at least make this an enum
    # note not to set both home isolation and full isolation true
    # whether to isolation detected agents to their homes (allow familial contact)
    home_isolation_sicks = False
    # whether to isolation detected agents fully (no contact)
    full_isolation_sicks = False
    # how many of the infected agents are actually caught and isolated
    caught_sicks_ratio = 0.3

    # policy stats
    # todo this reeeeally shouldn't be hard-coded
    # defines whether or not to apply a isolation (work shut-down)
    active_isolation = False
    # the date to stop work at
    stop_work_days = 30
    # the date to resume work at
    resume_work_days = 60

    # social stats
    # the average family size
    average_family_size = 5  # todo replace with distribution
    # the average workplace size
    average_work_size = 50  # todo replace with distribution
    # the average amount of stranger contacts per person
    average_amount_of_strangers = 200  # todo replace with distribution

    # relative strengths of each connection (in terms of infection chance)
    # todo so if all these strength are relative only to each other (and nothing else), whe are none of them 1?
    family_strength_not_workers = 0.75
    family_strength = 0.4
    work_strength = 0.04
    stranger_strength = 0.004

    @lru_cache
    def medical_state_machine(self):
        class SusceptibleTerminalState(SusceptibleState, TerminalState):
            pass

        class ImmuneStochasticState(ImmuneState, StochasticState):
            pass

        class ContagiousStochasticState(ContagiousState, StochasticState):
            pass

        class ImmuneTerminalState(ImmuneState, TerminalState):
            pass

        susceptible = SusceptibleTerminalState("Susceptible")
        latent = ImmuneStochasticState("Latent")
        silent = ContagiousStochasticState(
            "Silent", contagiousness=self.silent_infection_ratio
        )
        symptomatic = ContagiousStochasticState(
            "Symptomatic", contagiousness=self.symptomatic_infection_ratio
        )
        asymptomatic = ContagiousStochasticState(
            "Asymptomatic", contagiousness=self.asymptomatic_infection_ratio
        )

        hospitalized = ImmuneStochasticState("Hospitalized")
        icu = ImmuneStochasticState("ICU")

        deceased = ImmuneTerminalState("Deceased")
        recovered = ImmuneTerminalState("Recovered")

        ret = MedicalStateMachine(susceptible, latent)

        latent.add_transfer(silent, self.latent_to_silent_days, ...)

        silent.add_transfer(
            asymptomatic,
            self.silent_to_asymptomatic_days,
            self.silent_to_asymptomatic_probability,
        )
        silent.add_transfer(symptomatic, self.silent_to_symptomatic_days, ...)

        symptomatic.add_transfer(
            asymptomatic,
            self.symptomatic_to_asymptomatic_days,
            self.symptomatic_to_asymptomatic_probability,
        )
        symptomatic.add_transfer(
            hospitalized, self.symptomatic_to_hospitalized_days, ...
        )

        hospitalized.add_transfer(
            icu, self.hospitalized_to_icu_days, self.hospitalized_to_icu_probability
        )
        hospitalized.add_transfer(
            asymptomatic, self.hospitalized_to_asymptomatic_days, ...
        )

        icu.add_transfer(
            hospitalized,
            self.icu_to_hospitalized_days,
            self.icu_to_hospitalized_probability,
        )
        icu.add_transfer(deceased, self.icu_to_deceased_days, ...)

        asymptomatic.add_transfer(recovered, self.asymptomatic_to_recovered_days, ...)

        return ret


if __name__ == "__main__":
    c = Consts()
    print(c.average_time_in_each_state())
