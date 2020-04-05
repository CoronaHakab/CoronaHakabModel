from cached_property import cached_property
from generation.connection_types import ConnectionTypes
from util import rv_discrete


# todo switch to a named tuple, and allow loading from a file
class CirclesConsts:
    def __init__(self):
        # todo remove population size from main consts
        self.population_size = 10_000
        self.ages = [10, 40, 70]
        self.age_prob = [0.30, 0.45, 0.25]
        self.connection_type_prob_by_age_index = [
            {
                ConnectionTypes.Work: 0,
                ConnectionTypes.School: 0.95,
                ConnectionTypes.Family: 1.0,
                ConnectionTypes.Other: 1.0,
            },
            {
                ConnectionTypes.Work: 0.9,
                ConnectionTypes.School: 0,
                ConnectionTypes.Family: 1.0,
                ConnectionTypes.Other: 1.0,
            },
            {
                ConnectionTypes.Work: 0.25,
                ConnectionTypes.School: 0,
                ConnectionTypes.Family: 1.0,
                ConnectionTypes.Other: 1.0,
            },
        ]
        self.circle_size_distribution_by_connection_type = {
            ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
            ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
            ConnectionTypes.Family: ([1, 2, 3, 4, 5, 6, 7], [0.095, 0.227, 0.167, 0.184, 0.165, 0.081, 0.081]),
            ConnectionTypes.Other: ([100_000], [1.0]),
        }
        self.geo_circles_amount = 2
        self.geo_circles_names = ["north", "south"]
        self.geo_circles_agents_share = [0.6, 0.4]
        self.multi_zone_connection_type_to_geo_circle_probability = [
            {ConnectionTypes.Work: {self.geo_circles_names[0]: 0.7, self.geo_circles_names[1]: 0.3}},
            {ConnectionTypes.Work: {self.geo_circles_names[0]: 0.2, self.geo_circles_names[1]: 0.8}},
        ]

    @property
    def geographic_circles(self):
        assert self.geo_circles_amount == len(self.geo_circles_names)
        return [
            GeographicalCircleDataHolder(
                self.geo_circles_names[i],
                self.geo_circles_agents_share[i],
                self.age_distribution,
                self.circle_size_distribution_by_connection_type,
                self.connection_types_prob_by_age,
                self.multi_zone_connection_type_to_geo_circle_probability[i],
            )
            for i in range(self.geo_circles_amount)
        ]

    @cached_property
    def age_distribution(self):
        return rv_discrete(10, 70, values=(self.ages, self.age_prob))

    @cached_property
    def connection_types_prob_by_age(self):
        return {age: self.connection_type_prob_by_age_index[i] for i, age in enumerate(self.ages)}


class GeographicalCircleDataHolder:
    __slots__ = (
        "name",
        "agents_share",
        "age_distribution",
        "social_circles_logics",
        "connection_types_prob_by_age",
        "circles_size_distribution_by_connection_type",
        "multi_zone_connection_type_to_geo_circle_probability",
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
    ):
        self.name = name
        self.agents_share = agents_share
        self.age_distribution = age_distribution
        self.connection_types_prob_by_age = connection_types_prob_by_age
        self.circles_size_distribution_by_connection_type = circles_size_distribution_by_connection_type
        self.multi_zone_connection_type_to_geo_circle_probability = multi_zone_connection_type_to_geo_circle_probability
