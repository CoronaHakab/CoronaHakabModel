from dataclasses import dataclass
from generation.connection_types import ConnectionTypes


# todo fix type and allow importing from a json
@dataclass
class MatrixConsts:
    connection_type_to_connection_strength = {
        ConnectionTypes.Family: 3,
        ConnectionTypes.Work: 0.66,
        ConnectionTypes.School: 1,
        ConnectionTypes.Other: 0.23,
    }
    daily_connections_amount_by_connection_type = {
        ConnectionTypes.School: 6,
        ConnectionTypes.Work: 5.6,
        ConnectionTypes.Other: 0.4,
    }
    weekly_connections_amount_by_connection_type = {
        ConnectionTypes.School: 12.6,
        ConnectionTypes.Work: 12.6,
        ConnectionTypes.Other: 6,
    }
    use_parasymbolic_matrix = True
    clustering_switching_point = (50,)
    community_triad_probability = (1,)  # p
