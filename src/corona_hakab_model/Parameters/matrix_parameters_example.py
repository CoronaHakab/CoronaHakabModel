# flake8: noqa

# This is only as an example, duplicate this file and add the parameters you want changed

{
    "connection_type_to_connection_strength": {
        ConnectionTypes.Family: 3,
        ConnectionTypes.Work: 0.66,
        ConnectionTypes.School: 1,
        ConnectionTypes.Kindergarten: 1,
        ConnectionTypes.Other: 0.23,
    },
    "daily_connections_amount_by_connection_type": {
        ConnectionTypes.School: 6,
        ConnectionTypes.Kindergarten: 6,
        ConnectionTypes.Work: 5.6,
        ConnectionTypes.Other: 0.4,
    },
    "weekly_connections_amount_by_connection_type": {
        ConnectionTypes.School: 12.6,
        ConnectionTypes.Kindergarten: 12.6,
        ConnectionTypes.Work: 12.6,
        ConnectionTypes.Other: 6,
    },
    "use_parasymbolic_matrix": True,
    "clustering_switching_point": 50,
    "community_triad_probability": {
        ConnectionTypes.Other: 1,
        ConnectionTypes.Work: 1,
        ConnectionTypes.School: 1,
        ConnectionTypes.Kindergarten: 1
    },
}
