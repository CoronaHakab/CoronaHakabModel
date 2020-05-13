import os
from typing import Dict, List, NamedTuple
import jsonpickle

from generation.connection_types import ConnectionTypes
import numpy as np

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
    geo_circles_amount: int = 2
    geo_circles: List[Dict] = [
        {
            "name": "north",
            "ages": [10, 40, 70],
            "age_prob": [0.30, 0.45, 0.25],
            "teachers_workforce_ratio": 0.04,  # ratio of teachers out of workforce
            "kindergarten_workforce_ratio": 0.004,  # ratio of kindergarten workers out of workforce
            "connection_type_prob_by_age_index": [
                {
                    ConnectionTypes.Work: 0,
                    ConnectionTypes.Kindergarten: 0.1,
                    ConnectionTypes.School: 0.85,
                    ConnectionTypes.Family: 1.0,
                    ConnectionTypes.Other: 1.0,
                },
                {
                    ConnectionTypes.Work: 0.9,
                    ConnectionTypes.Kindergarten: 0,
                    ConnectionTypes.School: 0,
                    ConnectionTypes.Family: 1.0,
                    ConnectionTypes.Other: 1.0,
                },
                {
                    ConnectionTypes.Work: 0.25,
                    ConnectionTypes.Kindergarten: 0,
                    ConnectionTypes.School: 0,
                    ConnectionTypes.Family: 1.0,
                    ConnectionTypes.Other: 1.0,
                },
            ],
            "circle_size_distribution_by_connection_type": {
                ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
                ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
                ConnectionTypes.Kindergarten: ([10, 20], [0.5, 0.5]),
                ConnectionTypes.Family: ([1, 2, 3, 4, 5, 6, 7], [0.095, 0.227, 0.167, 0.184, 0.165, 0.081, 0.081]),
                ConnectionTypes.Other: ([100_000], [1.0]),
            },
        },
        {
            "name": "south",
            "ages": [10, 40, 70],
            "age_prob": [0.30, 0.45, 0.25],
            "teachers_workforce_ratio": 0.04,  # ratio of teachers out of workforce
            "kindergarten_workforce_ratio": 0.004,  # ratio of kindergarten workers out of workforce
            "connection_type_prob_by_age_index": [
                {
                    ConnectionTypes.Work: 0,
                    ConnectionTypes.Kindergarten: 0.1,
                    ConnectionTypes.School: 0.85,
                    ConnectionTypes.Family: 1.0,
                    ConnectionTypes.Other: 1.0,
                },
                {
                    ConnectionTypes.Work: 0.9,
                    ConnectionTypes.Kindergarten: 0,
                    ConnectionTypes.School: 0,
                    ConnectionTypes.Family: 1.0,
                    ConnectionTypes.Other: 1.0,
                },
                {
                    ConnectionTypes.Work: 0.25,
                    ConnectionTypes.Kindergarten: 0,
                    ConnectionTypes.School: 0,
                    ConnectionTypes.Family: 1.0,
                    ConnectionTypes.Other: 1.0,
                },
            ],
            "circle_size_distribution_by_connection_type": {
                ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
                ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
                ConnectionTypes.Kindergarten: ([10, 20], [0.5, 0.5]),
                ConnectionTypes.Family: ([1, 2, 3, 4, 5, 6, 7], [0.095, 0.227, 0.167, 0.184, 0.165, 0.081, 0.081]),
                ConnectionTypes.Other: ([100_000], [1.0]),
            },
        },
    ]
    random_connections_dist_mean: Dict = {
        ConnectionTypes.Work: 2.2,
        ConnectionTypes.School: 0.25,
        ConnectionTypes.Kindergarten: 0.25,
        ConnectionTypes.Other: 2.1,
    }
    random_connections_strength_factor: Dict = {
        ConnectionTypes.Work: 0.5,
        ConnectionTypes.School: 1.,
        ConnectionTypes.Kindergarten: 1.,
        ConnectionTypes.Other: 0.6,
    }
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

    def export(self, export_path, file_name: str = "circles_consts.json"):
        if not file_name.endswith(".json"):
            file_name += ".json"
        with open(os.path.join(export_path, file_name), "w") as export_file:
            export_file.write(jsonpickle.encode(self._asdict()))

    def get_geographic_circles(self):
        assert self.geo_circles_amount == len(self.geo_circles)
        return [
            GeographicalCircleDataHolder(
                geo_circle["name"],
                self.geo_circles_agents_share[i],
                self.get_age_distribution(geo_circle),
                geo_circle["circle_size_distribution_by_connection_type"],
                self.get_connection_types_prob_by_age(geo_circle),
                self.multi_zone_connection_type_to_geo_circle_probability[i],
                self.get_required_adult_distributions(geo_circle),
                geo_circle["teachers_workforce_ratio"],
                geo_circle["kindergarten_workforce_ratio"]
            )
            for i, geo_circle in enumerate(self.geo_circles)
        ]

    def get_age_distribution(self, geo_circle):
        return {"ages": geo_circle["ages"], "probs": geo_circle["age_prob"]}

    def get_connection_types_prob_by_age(self, geo_circle):
        return {age: geo_circle["connection_type_prob_by_age_index"][i] for i, age in enumerate(geo_circle["ages"])}

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def get_required_adult_distributions(self, geo_circle):
        """
        This function returns a dictionary of adult-children distributions (a numpy random distribution)
        for a ConnectionTypes and circle size.
        The returned data structure is a dictionary whose keys are ConnectionTypes.
        Each value is another dictionary relating circle size to the relevant random variable.
        The random distributions can return random variables (via dist.rvs()) that is the number of adults
        in the circle.  
        """
        all_students = 0
        kindergarten_children = 0
        all_teachers = 0
        kindergarten_workers = 0
        for i in range(len(geo_circle["ages"])):
            if geo_circle["ages"][i] <= 18:
                all_students += geo_circle["age_prob"][i] * geo_circle["connection_type_prob_by_age_index"][i][
                    ConnectionTypes.School]
                kindergarten_children += geo_circle["age_prob"][i] * geo_circle[
                    "connection_type_prob_by_age_index"][i][ConnectionTypes.Kindergarten]
            else:
                all_teachers += geo_circle["age_prob"][i] * geo_circle["connection_type_prob_by_age_index"][i][
                    ConnectionTypes.Work] * geo_circle["teachers_workforce_ratio"]
                kindergarten_workers += geo_circle["age_prob"][i] * geo_circle[
                    "connection_type_prob_by_age_index"][i][ConnectionTypes.Work] * geo_circle[
                                            "kindergarten_workforce_ratio"]

        # teacher to student ratio is the part of the school-going population that are teachers.
        # it is used to calculate how many teachers are needed in each school (according to school size)
        if all_students > 0:
            teacher_to_student_ratio = all_teachers / all_students
        else:
            teacher_to_student_ratio = 1

        # equivalently for kindergartens
        if kindergarten_children > 0:
            kindergarten_worker_to_child_ratio = kindergarten_workers / kindergarten_children
        else:
            kindergarten_worker_to_child_ratio = 1
        school_sizes = geo_circle["circle_size_distribution_by_connection_type"][ConnectionTypes.School][0]
        kindergarten_sizes = geo_circle["circle_size_distribution_by_connection_type"][ConnectionTypes.Kindergarten][0]

        family_sizes = geo_circle["circle_size_distribution_by_connection_type"][ConnectionTypes.Family][0]

        def const(value):
            return {'a': np.array([value])}

        def choices(values, probs):
            return {'a': np.array(values), 'p': np.array(probs)}

        family_distributions = dict()
        if 1 in family_sizes:
            family_distributions[1] = const(1)
        if 2 in family_sizes:
            family_distributions[2] = choices([1, 2], [0.2, 0.8])
        family_distributions.update(
            {size: choices([1, 2, 3], [0.1, 0.8, 0.1]) for size in family_sizes if size > 2})
        return {
            ConnectionTypes.School: {
                school_size: const(round(school_size * teacher_to_student_ratio))
                for school_size in school_sizes
            },
            ConnectionTypes.Kindergarten: {
                kindergarten_size: const(round(kindergarten_size * kindergarten_worker_to_child_ratio))
                for kindergarten_size in kindergarten_sizes
            },
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
        "adult_distributions",
        "teachers_workforce_ratio",
        "kindergarten_workforce_ratio"
    )

    # todo define how social circles logics should be represented
    def __init__(
            self,
            name: str,
            agents_share: float,
            age_distribution: Dict,
            circles_size_distribution_by_connection_type,
            connection_types_prob_by_age,
            multi_zone_connection_type_to_geo_circle_probability,
            adult_distributions,
            teachers_workforce_ratio,
            kindergarten_workforce_ratio
    ):
        self.name = name
        self.agents_share = agents_share
        self.age_distribution = age_distribution
        self.connection_types_prob_by_age = connection_types_prob_by_age
        self.circles_size_distribution_by_connection_type = circles_size_distribution_by_connection_type
        self.multi_zone_connection_type_to_geo_circle_probability = multi_zone_connection_type_to_geo_circle_probability
        self.adult_distributions = adult_distributions
        self.teachers_workforce_ratio = teachers_workforce_ratio
        self.kindergarten_workforce_ratio = kindergarten_workforce_ratio
