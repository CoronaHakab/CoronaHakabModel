import os
from typing import Dict, NamedTuple
import jsonpickle
import numpy as np
from dataclasses import dataclass
import math
from functools import lru_cache

from generation.connection_types import ConnectionTypes


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
        ConnectionTypes.Kindergarten: 1,
        ConnectionTypes.Other: 0.23,
    }
    daily_connections_amount_by_connection_type: Dict = {
        ConnectionTypes.School: 6,
        ConnectionTypes.Kindergarten: 6,
        ConnectionTypes.Work: 5.6,
        ConnectionTypes.Other: 0.4,
    }
    weekly_connections_amount_by_connection_type: Dict = {
        ConnectionTypes.School: 12.6,
        ConnectionTypes.Kindergarten: 12.6,
        ConnectionTypes.Work: 12.6,
        ConnectionTypes.Other: 6,
    }

    clustering_switching_point: int = 50  # TODO should this remain this way
    community_triad_probability: Dict = {
        ConnectionTypes.Other: 1,
        ConnectionTypes.Work: 1,
        ConnectionTypes.School: 1,
        ConnectionTypes.Kindergarten: 1
    }
    use_parasymbolic_matrix: bool = True

    @lru_cache(None)
    def get_connection_type_data(self, con_type: ConnectionTypes) -> "ConnectionTypeData":
        con_strength = self.connection_type_to_connection_strength[con_type]
        daily_amount = self.daily_connections_amount_by_connection_type[con_type] \
            if con_type in self.daily_connections_amount_by_connection_type else None
        weekly_amount = self.weekly_connections_amount_by_connection_type[con_type] \
            if con_type in self.weekly_connections_amount_by_connection_type else None
        triad_p = self.community_triad_probability[con_type] \
            if con_type in self.community_triad_probability else None

        return ConnectionTypeData(connection_type=con_type, connection_strength=con_strength, triad_p=triad_p,
                                  daily_connections_amount=daily_amount, weekly_connections_amount=weekly_amount)

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

    def export(self, export_path, file_name: str = "matrix_consts.json"):
        if not file_name.endswith(".json"):
            file_name += ".json"
        with open(os.path.join(export_path, file_name), "w") as export_file:
            export_file.write(jsonpickle.encode(self._asdict()))

    # overriding hash and eq to allow caching while using un-hashable attributes
    __hash__ = object.__hash__
    __eq__ = object.__eq__


@dataclass
class ConnectionTypeData:
    """holds all the connection type data for better code readability
    will gain importance when the amount of parameters will increase"""

    connection_type: ConnectionTypes
    connection_strength: float
    # defaults to none cause not all connection types has it
    daily_connections_amount: float = None
    weekly_connections_amount: float = None
    triad_p: float = None

    def get_rounded_connections_amount(self, shape: int) -> np.ndarray:
        """returns the total amount of connections for the given type
        randomly chooses between floor and ceil such that the average will be correct
        raises an error if daily or weekly connections amount is undefined"""

        assert self.daily_connections_amount is not None and self.weekly_connections_amount is not None, \
            "rolled daily or weekly connection on a type without daily or weekly connections"
        total_connections = self.total_connections_amount
        if total_connections % 1 == 0:
            return np.array([int(total_connections)] * shape)

        floor_prob = math.ceil(total_connections) - total_connections
        ceil_prob = total_connections - math.floor(total_connections)

        return np.random.choice([math.floor(total_connections), math.ceil(total_connections)], size=shape, p=[floor_prob, ceil_prob])

    @property
    def total_connections_amount(self) -> float:
        assert self.daily_connections_amount is not None and self.weekly_connections_amount is not None, \
            "asked for total connections amount when either daily or weekly connections amount is undefined"
        return self.daily_connections_amount + self.weekly_connections_amount

    def get_strengths(self, shape: int = 1) -> np.ndarray:
        """for each meeting, rolls whether is daily or weekly, and chooses the strength accordingly"""

        if self.daily_connections_amount is None or self.weekly_connections_amount is None:
            return np.full(shape=shape, fill_value=self.connection_strength)
        if self.total_connections_amount == 0: # flag for easy narrow down of the simulation
            return np.full(shape=shape, fill_value=0.0)
        else:
            # rolls for each connection, whether it is daily or weekly
            daily_share = self.daily_connections_amount / self.total_connections_amount
            weekly_share = self.weekly_connections_amount / self.total_connections_amount

            return np.random.choice(
                [self.connection_strength, self.connection_strength / 7], size=shape, p=[daily_share, weekly_share]
            )