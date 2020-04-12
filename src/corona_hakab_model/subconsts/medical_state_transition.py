from dataclasses import dataclass
from typing import NamedTuple, Dict

from scipy.stats import rv_discrete

from util import dist


@dataclass
class MedicalStateTransition:
    @dataclass
    class DayDistributions:
        # Tsvika: Currently the distribution is selected based on the number of input parameters.
        # Think we should do something more readable later on.
        # For example: "latent_to_silent_days": {"type":"uniform","lower_bound":1,"upper_bound":3}
        # disease states transition lengths distributions
        latent_to_silent: rv_discrete = dist(1, 3)
        silent_to_asymptomatic: rv_discrete = dist(0, 3, 10)
        silent_to_symptomatic: rv_discrete = dist(0, 3, 10)
        asymptomatic_to_recovered: rv_discrete = dist(3, 5, 7)
        symptomatic_to_asymptomatic: rv_discrete = dist(7, 10, 14)
        symptomatic_to_hospitalized: rv_discrete = dist(0, 1.5, 10)
        hospitalized_to_asymptomatic: rv_discrete = dist(18)
        hospitalized_to_icu: rv_discrete = dist(5)
        icu_to_deceased: rv_discrete = dist(7)
        icu_to_hospitalized: rv_discrete = dist(7)

        @classmethod
        def json_dict_to_instance(cls, **kwargs):
            self = cls(**kwargs)
            for key, value_dict in kwargs.items():
                if "dist" in value_dict:
                    setattr(self, key, dist(*value_dict["dist"]))
            return self

    class TransitionProbabilities(NamedTuple):
        # average probability for transitions:
        silent_to_asymptomatic: float = 0.2
        symptomatic_to_asymptomatic: float = 0.85
        hospitalized_to_asymptomatic: float = 0.8
        icu_to_hospitalized: float = 0.65

        @property
        def silent_to_symptomatic(self) -> float:
            return 1 - self.silent_to_asymptomatic

        @property
        def symptomatic_to_hospitalized(self) -> float:
            return 1 - self.symptomatic_to_asymptomatic

        @property
        def hospitalized_to_icu(self) -> float:
            return 1 - self.hospitalized_to_asymptomatic

        @property
        def icu_to_dead(self) -> float:
            return 1 - self.icu_to_hospitalized

    @classmethod
    def json_dict_to_instance(cls, **kwargs):
        day_distributions = kwargs.pop('day_distributions')
        transition_probabilities = kwargs.pop('transition_probabilities')
        self = cls()
        self.day_distributions = cls.day_distributions.json_dict_to_instance(**day_distributions)
        self.transition_probabilities = cls.TransitionProbabilities(**transition_probabilities)
        return self

    day_distributions = DayDistributions()
    transition_probabilities = TransitionProbabilities()
