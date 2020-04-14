from typing import Dict, NamedTuple, Tuple

from generation.connection_types import ConnectionTypes
from util import rv_discrete


"""
Overview:

MatrixConsts class is a named tuple consts for the Matrix creation stage of the SimulationData generation.
it may either be made using default params, or by loading parameters from a file.
Usage:
1. Create a default consts object - consts = Consts()
2. Load a parameters file - consts = Consts.from_file(path)
"""


class MatrixConsts(NamedTuple):
    # In order to create instances of the class the fields must be typed

    # Attributes and default values

    # Connections Weights
    connection_type_to_const_weight: Dict = {
        ConnectionTypes.Family: 3,
    }
    daily_connection_type_to_weight_generator: Dict = {
        ConnectionTypes.Work: rv_discrete(values=([0.1, 0.2, 0.6, 2.5, 4], [0.12, 0.18, 0.2, 0.19, 0.31])),
        ConnectionTypes.School: rv_discrete(values=([0.1, 0.2, 0.6, 2.5, 4], [0.06, 0.07, 0.11, 0.25, 0.51])),
        ConnectionTypes.Other: rv_discrete(values=([0.1, 0.2, 0.6, 2.5, 4], [0.21, 0.26, 0.19, 0.23, 0.11])),
    }
    weekly_connection_type_to_weight_generator: Dict = {
        ConnectionTypes.Work: rv_discrete(values=([0.1, 0.2, 0.6, 2.5, 4], [0.22, 0.23, 0.23, 0.18, 0.14])),
        ConnectionTypes.School: rv_discrete(values=([0.1, 0.2, 0.6, 2.5, 4], [0.1, 0.15, 0.24, 0.32, 0.19])),
        ConnectionTypes.Other: rv_discrete(values=([0.1, 0.2, 0.6, 2.5, 4], [0.22, 0.26, 0.24, 0.22, 0.06])),
    }
    daily_connection_type_to_probability_generator: Dict = {
        ConnectionTypes.Work: rv_discrete(values=([1], [1.0])),
        ConnectionTypes.School: rv_discrete(values=([1], [1.0])),
        ConnectionTypes.Other: rv_discrete(values=([1], [1.0])),
    }
    weekly_connection_type_to_probability_generator: Dict = {
        ConnectionTypes.Work: rv_discrete(values=([1/7], [1.0])),
        ConnectionTypes.School: rv_discrete(values=([1/7], [1.0])),
        ConnectionTypes.Other: rv_discrete(values=([1/7], [1.0])),
    }
    daily_connections_amount_by_connection_type: Dict = {
        ConnectionTypes.School: 6,
        ConnectionTypes.Work: 5.6,
        ConnectionTypes.Other: 0.4,
    }
    weekly_connections_amount_by_connection_type: Dict = {
        ConnectionTypes.School: 12.6,
        ConnectionTypes.Work: 12.6,
        ConnectionTypes.Other: 6,
    }
    # todo switch to the real magic operator, or a basic one for tests
    connection_type_to_magic_operator: Dict = {
        ConnectionTypes.Family: None,
        ConnectionTypes.School: None,
        ConnectionTypes.Work: None,
        ConnectionTypes.Other: None
    }

    clustering_switching_point: int = 50
    community_triad_probability: Dict = {
        ConnectionTypes.School: 1,
        ConnectionTypes.Work: 1,
        ConnectionTypes.Other: 1,
    }
    use_parasymbolic_matrix: bool = True

    @classmethod
    def from_file(cls, param_path):
        """
        Load parameters from file and return MatrixConsts object with those values.

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

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__

