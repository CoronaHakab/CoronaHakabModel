from typing import Dict, List, NamedTuple
from generation.connection_types import ConnectionTypes
from util import rv_discrete, randint


"""
Overview:

CirclesConsts class is a named tuple consts for the Circles creation stage of the SimulationData generation.
it may either be made using default params, or by loading parameters from a file.
Usage:
1. Create a default consts object - consts = Consts()
2. Load a parameters file - consts = Consts.from_file(path)
"""


class CirclesConsts(NamedTuple):
    population_size: int = 20_000
    ages: List[int] = [10, 40, 70]
    age_prob: List[int] = [0.30, 0.45, 0.25]
    teachers_ratio = 0.04 # ratio of teachers out of workforce
    connection_type_prob_by_age_index: List = [
        {
            ConnectionTypes.Work: 0,
            ConnectionTypes.School: 0.95,
            ConnectionTypes.Family: 1.0,
            ConnectionTypes.Other: 1.0,
        },
        {
            ConnectionTypes.Work: 0.9 * (1 - teachers_ratio),
            ConnectionTypes.School: 0.9 * teachers_ratio,
            ConnectionTypes.Family: 1.0,
            ConnectionTypes.Other: 1.0,
        },
        {
            ConnectionTypes.Work: 0.25 * (1 - teachers_ratio),
            ConnectionTypes.School: 0.25 * teachers_ratio,
            ConnectionTypes.Family: 1.0,
            ConnectionTypes.Other: 1.0,
        },
    ]
    circle_size_distribution_by_connection_type: Dict = {
        ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
        ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
        ConnectionTypes.Family: ([1, 2, 3, 4, 5, 6, 7], [0.095, 0.227, 0.167, 0.184, 0.165, 0.081, 0.081]),
        ConnectionTypes.Other: ([100_000], [1.0]),
    }
    geo_circles_amount: int = 2
    geo_circles_names: List[str] = ["north", "south"]
    geo_circles_agents_share: List[float] = [0.6, 0.4]
    multi_zone_connection_type_to_geo_circle_probability: List = [
        {ConnectionTypes.Work: {"north": 0.7, "south": 0.3}},
        {ConnectionTypes.Work: {"north": 0.2, "south": 0.8}},
    ]

    @classmethod
    def from_file(cls, param_path):
        """
        Load parameters from file and return CirclesConsts object with those values.

        No need to sanitize the eval'd data as we disabled __builtins__ and only passed specific functions
        """
        with open(param_path, "rt") as read_file:
            data = read_file.read()

        # expressions to evaluate
        expressions = {
            "__builtins__": None,
            "ConnectionTypes": ConnectionTypes,
        }

        parameters = eval(data, expressions)

        return cls(**parameters)

    def get_geographic_circles(self):
        assert self.geo_circles_amount == len(self.geo_circles_names)
        return [
            GeographicalCircleDataHolder(
                self.geo_circles_names[i],
                self.geo_circles_agents_share[i],
                self.get_age_distribution(),
                self.circle_size_distribution_by_connection_type,
                self.get_connection_types_prob_by_age(),
                self.multi_zone_connection_type_to_geo_circle_probability[i],
                self.get_required_adult_distributions()
            )
            for i in range(self.geo_circles_amount)
        ]

    def get_age_distribution(self):
        return rv_discrete(10, 70, values=(self.ages, self.age_prob))

    def get_connection_types_prob_by_age(self):
        return {age: self.connection_type_prob_by_age_index[i] for i, age in enumerate(self.ages)}
    
    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__


    def get_required_adult_distributions(self):
        students = 0
        teachers = 0
        for i in range(self.ages):
            if self.ages[i] <= 18:
                students += self.age_prob[i]*self.connection_type_prob_by_age_index[i][ConnectionTypes.School]
            else:
                teachers += self.age_prob[i]*self.connection_type_prob_by_age_index[i][ConnectionTypes.School]
                
        teacher_student_ratio = teachers * 1.0 / students
        school_sizes = self.circle_size_distribution_by_connection_type[ConnectionTypes.School][0]
        
        family_sizes = self.circle_size_distribution_by_connection_type[ConnectionTypes.Family][0]
        family_distributions = {size: randint(1,2) for size in family_sizes if size==1}
        family_distributions.update({size: rv_discrete(1, 2, values=([1, 2], [0.2, 0.8])) for size in family_sizes if size==2})
        family_distributions.update({size: rv_discrete(1, 3, values=([1, 2, 3], [0.1, 0.8, 0.1])) for size in family_sizes if size > 2})
        return {
            ConnectionTypes.School: {school_size: randint(round(school_size*teacher_student_ratio), round(school_size*teacher_student_ratio)+1) for school_size in school_sizes},      
            ConnectionTypes.Family: family_distributions
        }


class GeographicalCircleDataHolder:
    __slots__ = (
        "name",
        "agents_share",
        "age_distribution",
        "social_circles_logics",
        "connection_types_prob_by_age",
        "circles_size_distribution_by_connection_type",
        "multi_zone_connection_type_to_geo_circle_probability",
        "adult_distributions"
    )

    # todo define how social circles logics should be represented
    def __init__(
        self,
        name: str,
        agents_share: float,
        age_distribution: rv_discrete,
        circles_size_distribution_by_connection_type,
        connection_types_prob_by_age,
        multi_zone_connection_type_to_geo_circle_probability,
        adult_distributions        
    ):
        self.name = name
        self.agents_share = agents_share
        self.age_distribution = age_distribution
        self.connection_types_prob_by_age = connection_types_prob_by_age
        self.circles_size_distribution_by_connection_type = circles_size_distribution_by_connection_type
        self.multi_zone_connection_type_to_geo_circle_probability = multi_zone_connection_type_to_geo_circle_probability
        self.adult_distributions = adult_distributions