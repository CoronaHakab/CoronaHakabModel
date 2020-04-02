from collections import namedtuple
from util import dist, rv_discrete

default_parameters = {
    "population_size": 10_000,
    "age_distribution": rv_discrete(10, 70, values=([10, 40, 70], [0.30, 0.45, 0.25]))
}

ConstParameters = namedtuple(
    "ConstParameters",
    sorted(default_parameters),
    defaults=[default_parameters[key] for key in sorted(default_parameters)],
)


class GenerationConsts(ConstParameters):
    __slots__ = ()

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
        GenerationConsts.sanitize_parameters(parameters)

        return GenerationConsts(**parameters)

    @staticmethod
    def sanitize_parameters(parameters):
        consts = GenerationConsts(**parameters)
        try:
            hash(consts)
        except TypeError as e:
            raise TypeError("Unhashable value in parameters") from e

    @property
    def geographic_circles(self):
        return [
            GeographicalCircleDataHolder(0.5, self.age_distribution)
        ]

class GeographicalCircleDataHolder:
    __slots__ = "agents_share", "age_distribution", "social_circles_logics"

    # todo define how social circles logics should be represented
    def __init__(self, agents_share: float, age_distribution: rv_discrete = None, social_circles_logics=None):
        self.agents_share = agents_share
        self. age_distribution = age_distribution
        self.social_circles_logics = social_circles_logics
