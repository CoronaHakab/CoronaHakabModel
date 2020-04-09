from generation.connection_types import ConnectionTypes
from typing import Dict, NamedTuple, Tuple

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

    # Strengths of each connection type - how likely are people to infect one another in that circle
    connection_type_to_connection_strength: Dict = {
        ConnectionTypes.Family: 3,
        ConnectionTypes.Work: 0.66,
        ConnectionTypes.School: 1,
        ConnectionTypes.Other: 0.23,
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

    clustering_switching_point: Tuple = (50,)
    community_triad_probability: Tuple = (1,)
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
