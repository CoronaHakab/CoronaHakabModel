from typing import NamedTuple


class Consts(NamedTuple):
    # corona stats
    # average time between infection and entering symptomatic or asymptomatic stages
    average_silent_time_days: int = 1  # todo replace with distribution
    # average time an infected is sick since becoming symptomatic or asymptomatic
    average_sick_time_days: int = 3  # todo replace with distribution

    # probability of sick agent (symptomatic or asymptomatic) dying per day
    death_ratio: float = 0.05  # todo change to a per day-basis?
    # probability of an infected agent to become asymptomatic
    Asymptomatic_ratio: float = 0.5
    # probability of an infected symptomatic agent infecting others
    Symptomatic_infection_ratio: float = 0.75
    # probability of an infected asymptomatic agent infecting others
    ASymptomatic_infection_ratio: float = 0.25
    # base r0 of the disease
    r0: float = 1.5

    def expected_infection_ratio(self):
        """
        The expected infection ratio of a random infected agent
        """
        return self.ASymptomatic_infection_ratio * self.Asymptomatic_ratio \
               + self.Symptomatic_infection_ratio * (1 - self.Asymptomatic_ratio)

    # simulation parameters
    population_size = 10_000
    total_steps = 250
    initial_infected_count = 20

    # quarantine policy
    # todo why does this exist? doesn't the policy set this? at least make this an enum
    # note not to set both home quarantine and full quarantine true
    # whether to quarantine detected agents to their homes (allow familial contact)
    home_quarantine_sicks = False
    # whether to quarantine detected agents fully (no contact)
    full_quarantine_sicks = False
    # how many of the infected agents are actually caught and quarantined
    caught_sicks_ratio = 0.2

    # policy stats
    # todo this reeeeally shouldn't be hard-coded
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

    # relative strangths of each connection (in terms of infection chance)
    # todo so if all these strength are relative only to each other (and nothing else), whe are none of them 1?
    family_strength_not_workers = 0.75
    family_strength = 0.4
    work_strength = 0.04
    stranger_strength = 0.004
