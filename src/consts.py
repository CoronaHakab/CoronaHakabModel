from typing import NamedTuple


class Consts(NamedTuple):
    # corona stats
    average_silent_time_days: int = 1  # todo replace with distribution
    average_sick_time_days: int = 3  # todo replace with distribution

    death_ratio: float = 0.05
    Asymptomatic_ratio: float = 0.5
    Symptomatic_infection_ratio: float = 0.75
    ASymptomatic_infection_ratio: float = 0.25
    r0: float = 1.5

    def expected_infection_ratio(self):
        return self.ASymptomatic_infection_ratio * self.Asymptomatic_ratio \
               + self.Symptomatic_infection_ratio * (1 - self.Asymptomatic_ratio)

    # simulation parameters
    SIZE_OF_POPULATION = 10_000
    STEPS_TO_RUN = 250
    INITIAL_INFECTED_COUNT = 20

    # quarantine policy
    # todo why does this exist? doesn't the policy set this?
    home_quarantine_sicks = False  # note not to set both home quarantine and full quarantine true
    full_quarantine_sicks = False
    caught_sicks_ratio = 0.2

    # policy stats
    # todo this reeeeally shouldn't be hard-coded
    stop_work_days = 30
    resume_work_days = 60

    # social stats
    average_family_size = 5  # todo replace with distribution
    average_work_size = 50  # todo replace with distribution
    average_amount_of_strangers = 200  # todo replace with distribution
    # todo so if all these strength are relative only to each other (and nothing else), whe are none of them 1?
    family_strength_not_workers = 0.75
    family_strength = 0.4
    work_strength = 0.04
    stranger_strength = 0.004
