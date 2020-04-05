from generation.connection_types import ConnectionTypes
from collections import namedtuple
from typing import NamedTuple


# todo switch with a named tuple and allow import / export
class MatrixConsts(NamedTuple):
    connection_type_to_connection_strength = {
        ConnectionTypes.Family: 1,
        ConnectionTypes.Work: 0.2,
        ConnectionTypes.School: 0.3,
        ConnectionTypes.Other: 0.05
    }
    daily_connections_amount_by_connection_type = {
        ConnectionTypes.School: 6,
        ConnectionTypes.Work: 5.6,
        ConnectionTypes.Other: 0.4
    }
    weekly_connections_amount_by_connection_type = {
        ConnectionTypes.School: 12.6,
        ConnectionTypes.Work: 12.6,
        ConnectionTypes.Other: 6
    }
    use_parasimbolic_matrix = True,
    clustering_switching_point = 20,
    community_member_edges = 2, #m
    community_triad_probability = 0.8, #p


