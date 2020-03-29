# flake8: noqa

#This is only as an example, and holds the default values


{
    "population_size": 10_000,
    "total_steps": 350,
    "initial_infected_count": 20,
    # Tsvika: Currently the distribution is selected based on the number of input parameters.
    # Think we should do something more readable later on.
    # For example: "latent_to_silent_days": {"type":"uniform","lower_bound":1,"upper_bound":3}
    "latent_to_silent_days": dist(1, 3),
    "silent_to_asymptomatic_days": dist(0, 3, 10),
    "silent_to_symptomatic_days": dist(0, 3, 10),
    "asymptomatic_to_recovered_days": dist(3, 5, 7),
    "symptomatic_to_asymptomatic_days": dist(7, 10, 14),
    "symptomatic_to_hospitalized_days": dist(
        0, 1.5, 10
    ),  # todo range not specified in sources
    "hospitalized_to_asymptomatic_days": dist(18),
    "hospitalized_to_icu_days": dist(5),  # todo probably has a range
    "icu_to_deceased_days": dist(7),  # todo probably has a range
    "icu_to_hospitalized_days": dist(
        7
    ),  # todo maybe the program should juts print a question mark,  we'll see how the researchers like that!
    # average probability for transmitions:
    "silent_to_asymptomatic_probability": 0.2,
    "symptomatic_to_asymptomatic_probability": 0.85,
    "hospitalized_to_asymptomatic_probability": 0.8,
    "icu_to_hospitalized_probability": 0.65,
    # probability of an infected symptomatic agent infecting others
    "symptomatic_infection_ratio": 0.75,
    # probability of an infected asymptomatic agent infecting others
    "asymptomatic_infection_ratio": 0.25,
    # probability of an infected silent agent infecting others
    "silent_infection_ratio": 0.3,  # todo i made this up, need to get the real number
    # base r0 of the disease
    "r0": 2.4,
    # isolation policy
    # todo why does this exist? doesn't the policy set this? at least make this an enum
    # note not to set both home isolation and full isolation true
    # whether to isolation detected agents to their homes (allow familial contact)
    "home_isolation_sicks": False,
    # whether to isolation detected agents fully (no contact)
    "full_isolation_sicks": False,
    # how many of the infected agents are actually caught and isolated
    "caught_sicks_ratio": 0.3,
    # policy stats
    # todo this reeeeally shouldn't be hard-coded
    # defines whether or not to apply a isolation (work shut-down)
    "active_isolation": False,
    # the date to stop work at
    "stop_work_days": 30,
    # the date to resume work at
    "resume_work_days": 60,
    # social stats
    # the average family size
    "average_family_size": 5,  # todo replace with distribution
    # the average workplace size
    "average_work_size": 50,  # todo replace with distribution
    # the average amount of stranger contacts per person
    "average_amount_of_strangers": 200,  # todo replace with distribution
    # relative strengths of each connection (in terms of infection chance)
    # todo so if all these strength are relative only to each other (and nothing else), whe are none of them 1?
    "family_strength_not_workers": 0.75,
    "family_strength": 0.4,
    "work_strength": 0.04,
    "stranger_strength": 0.004,
}
