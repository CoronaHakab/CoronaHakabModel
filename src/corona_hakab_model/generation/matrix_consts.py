from collections import namedtuple
from generation.connection_types import ConnectionTypes

"""
Overview:

We have default_parameters - it is our template ans as the name suggests, holds the default values
Using that template, we create MatrixConsts, a named tuple.
the Consts class inherits from MatrixConsts the fields, and adds the methods.
This is how we preserve the efficiency of a NamedTuple but also get dynamic values
Usage:
1. Create a default consts object - consts = MatrixConsts()
2. Load a parameters file - consts = MatrixConsts.from_file(path)
"""
default_parameters = {
    "connection_type_to_connection_strength": {
        ConnectionTypes.Family: 3,
        ConnectionTypes.Work: 0.66,
        ConnectionTypes.School: 1,
        ConnectionTypes.Other: 0.23,
    },
    "daily_connections_amount_by_connection_type": {
        ConnectionTypes.School: 6,
        ConnectionTypes.Work: 5.6,
        ConnectionTypes.Other: 0.4,
    },
    "weekly_connections_amount_by_connection_type": {
        ConnectionTypes.School: 12.6,
        ConnectionTypes.Work: 12.6,
        ConnectionTypes.Other: 6,
    },
    "use_parasymbolic_matrix": True,
    "clustering_switching_point": (50,),
    "community_triad_probability": (1,),
}

MatrixParameters = namedtuple(
    "MatrixParameters",
    sorted(default_parameters),
    defaults=[default_parameters[key] for key in sorted(default_parameters)],
)


class MatrixConsts(MatrixParameters):
    __slots__ = ()

    @staticmethod
    def from_file(param_path):
        """
        Load parameters from file and return MatrixConsts object with those values.

        No need to sanitize the eval'd data as we disabled __builtins__ and only passed specific functions
        Documentation about what is allowed and not allowed can be found at the top of this page.
        """
        with open(param_path, "rt") as read_file:
            data = read_file.read()

        parameters = eval(data, {"__builtins__": None, "ConnectionTypes": ConnectionTypes})

        return MatrixConsts(**parameters)
