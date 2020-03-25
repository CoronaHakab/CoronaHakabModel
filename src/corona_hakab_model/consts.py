from functools import lru_cache
from itertools import count
from math import fsum, isclose
from typing import NamedTuple

import numpy as np
from scipy.stats import rv_discrete

from medical_state import InfectableState, InfectiousState, ImmuneState
from medical_state_machine import MedicalStateMachine
from state_machine import TerminalState, StochasticState
from util import dist


class Consts(NamedTuple):
    # simulation parameters
    population_size = 10_000
    total_steps = 200
    initial_infected_count = 20

    # corona stats
    # todo replace with distribution
    # average state mechine transmitions times:
    latent_to_silent_days: rv_discrete = dist(1, 3)
    silent_to_asymptomatic_days: rv_discrete = dist(0, 3, 10)
    silent_to_symptomatic_days: rv_discrete = dist(0, 3, 10)
    asymptomatic_to_recovered_days: rv_discrete = dist(3, 5, 7)
    symptomatic_to_asymptomatic_days: rv_discrete = dist(7, 10, 14)
    symptomatic_to_hospitalized_days: rv_discrete = dist(0, 1.5, 10)  # todo range not specified in sources
    hospitalized_to_asymptomatic_days: rv_discrete = dist(18)
    hospitalized_to_icu_days: rv_discrete = dist(5)  # todo probably has a range
    icu_to_deceased_days: rv_discrete = dist(7)  # todo probably has a range
    icu_to_hospitalized_days: rv_discrete = dist(7)  # todo maybe the program should juts print a question mark,

    # we'll see how the researchers like that!

    @lru_cache()
    def n_average_infecting_days(self):
        # todo fix
        """
        returns the expected time of infectivness of an infected people (for normalization)
        assuming you are not contagious when in a hospital nor in icu.
        also ignoring moving back from icu to asymptomatic
        """

        per_TOL = 1e-6
        p_TOL = 1e-2
        min_t = 10

        m = self.medical_state_machine()
        i_state = m.state_upon_infection
        infectious_states = [s for s in m.states if s.infectiousness]
        infectious_arr = []
        for t in count():
            infectious_arr.append(
                v := fsum(
                    i_state.probability(t, s, per_TOL) for s in infectious_states
                )
            )
            if t > min_t and v < p_TOL:
                break

        infectious_arr = np.array(infectious_arr)
        return np.sum(np.arange(len(infectious_arr)-1) * infectious_arr[:-1] * (1-infectious_arr[1:]))

    def average_infecting_days(self):
        """
        returns the expected time of infectivness of an infected people (for normalization)
        assuming you are not contagious when in a hospital nor in icu.
        also ignoring moving back from icu to asymptomatic
        """
        silent_time = (
                self.silent_to_asymptomatic_probability
                * self.silent_to_asymptomatic_days.mean()
                + self.silent_to_symptomatic_probability
                * self.silent_to_symptomatic_days.mean()
        )
        asymptomatic_time = (
                self.asymptomatic_to_recovered_days.mean()
                * self.silent_to_asymptomatic_probability
        )
        symptomatic_time = self.silent_to_symptomatic_probability * (
                (self.symptomatic_to_asymptomatic_days.mean() + asymptomatic_time)
                * self.symptomatic_to_asymptomatic_probability
                + self.symptomatic_to_hospitalized_days.mean()
                * self.symptomatic_to_hospitalized_probability
        )
        hosplital_time = (
                self.silent_to_symptomatic_probability
                * self.symptomatic_to_hospitalized_probability
                * self.hospitalized_to_asymptomatic_probability
                * asymptomatic_time
        )
        return silent_time + asymptomatic_time + symptomatic_time + hosplital_time

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

    def expected_infection_ratio(self):
        """
        The expected infection ratio of a random infected agent
        """
        asymptomatic_time = (
                self.asymptomatic_to_recovered_days.mean()
                * self.silent_to_asymptomatic_probability
        )
        symptomatic_time = self.silent_to_symptomatic_probability * (
                self.symptomatic_to_asymptomatic_days.mean()
                * self.symptomatic_to_asymptomatic_probability
                + self.symptomatic_to_hospitalized_days.mean()
                * self.symptomatic_to_hospitalized_probability
        )
        silent_time = (
                self.silent_to_symptomatic_probability
                * self.silent_to_symptomatic_days.mean()
                + self.silent_to_asymptomatic_probability
                * self.silent_to_asymptomatic_days.mean()
        )
        total_time = asymptomatic_time + symptomatic_time + silent_time
        return (
                       self.asymptomatic_infection_ratio * asymptomatic_time
                       + self.symptomatic_infection_ratio * symptomatic_time
                       + self.silent_infection_ratio * silent_time
               ) / total_time

    # quarantine policy
    # todo why does this exist? doesn't the policy set this? at least make this an enum
    # note not to set both home quarantine and full quarantine true
    # whether to quarantine detected agents to their homes (allow familial contact)
    home_quarantine_sicks = False
    # whether to quarantine detected agents fully (no contact)
    full_quarantine_sicks = False
    # how many of the infected agents are actually caught and quarantined
    caught_sicks_ratio = 0.3

    # policy stats
    # todo this reeeeally shouldn't be hard-coded
    # defines whether or not to apply a quarantine (work shut-down)
    active_quarantine = False
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
        class InfectableTerminalState(InfectableState, TerminalState):
            pass

        class ImmuneStochasticState(ImmuneState, StochasticState):
            pass

        class InfectiousStochasticState(InfectiousState, StochasticState):
            pass

        class ImmuneTerminalState(ImmuneState, TerminalState):
            pass

        susceptible = InfectableTerminalState("Susceptible")
        latent = ImmuneStochasticState("Latent")
        silent = InfectiousStochasticState("Silent", infectiousness=self.silent_infection_ratio)
        symptomatic = InfectiousStochasticState("Symptomatic", infectiousness=self.symptomatic_infection_ratio)
        asymptomatic = InfectiousStochasticState("Asymptomatic", infectiousness=self.asymptomatic_infection_ratio)

        hospitalized = ImmuneStochasticState("Hospitalized")
        icu = ImmuneStochasticState("ICU")

        deceased = ImmuneTerminalState("Deceased")
        recovered = ImmuneTerminalState("Recovered")

        ret = MedicalStateMachine(susceptible, latent)

        latent.add_transfer(silent, self.latent_to_silent_days, ...)

        silent.add_transfer(asymptomatic, self.silent_to_asymptomatic_days, self.silent_to_asymptomatic_probability)
        silent.add_transfer(symptomatic, self.silent_to_symptomatic_days, ...)

        symptomatic.add_transfer(asymptomatic, self.symptomatic_to_asymptomatic_days,
                                 self.symptomatic_to_asymptomatic_probability)
        symptomatic.add_transfer(hospitalized, self.symptomatic_to_hospitalized_days, ...)

        hospitalized.add_transfer(icu, self.hospitalized_to_icu_days, self.hospitalized_to_icu_probability)
        hospitalized.add_transfer(asymptomatic, self.hospitalized_to_asymptomatic_days, ...)

        icu.add_transfer(hospitalized, self.icu_to_hospitalized_days, self.icu_to_hospitalized_probability)
        icu.add_transfer(deceased, self.icu_to_deceased_days, ...)

        asymptomatic.add_transfer(recovered, self.asymptomatic_to_recovered_days, ...)

        return ret

if __name__ == '__main__':
    c = Consts()
    print(c.average_infecting_days())
    print(c.n_average_infecting_days())