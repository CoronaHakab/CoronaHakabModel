# flake8: noqa

# This is only as an example, duplicate this file and add the parameters you want changed

{
    "population_size": 20_000,
    "geo_circles_amount": 2,
    "geo_circles": [
        {
            "name": "north",
            "ages": [10, 40, 70],
            "age_prob": [0.30, 0.45, 0.25],
            "teachers_workforce_ratio": 0.04,  # ratio of teachers out of workforce
            "connection_type_prob_by_age_index": [
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
            ],
            "circle_size_distribution_by_connection_type": {
                ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
                ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
                ConnectionTypes.Family: ([1, 2, 3, 4, 5, 6, 7], [0.095, 0.227, 0.167, 0.184, 0.165, 0.081, 0.081]),
                ConnectionTypes.Other: ([100_000], [1.0]),
            },
        },
        {
            "name": "south",
            "ages": [10, 40, 70],
            "age_prob": [0.30, 0.45, 0.25],
            "teachers_workforce_ratio": 0.04,  # ratio of teachers out of workforce
            "connection_type_prob_by_age_index": [
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
            ],
            "circle_size_distribution_by_connection_type": {
                ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
                ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
                ConnectionTypes.Family: ([1, 2, 3, 4, 5, 6, 7], [0.095, 0.227, 0.167, 0.184, 0.165, 0.081, 0.081]),
                ConnectionTypes.Other: ([100_000], [1.0]),
            },
        },
    ],
    "geo_circles_agents_share": [0.6, 0.4],
    "multi_zone_connection_type_to_geo_circle_probability": [
        {ConnectionTypes.Work: {"north": 0.7, "south": 0.3}},
        {ConnectionTypes.Work: {"north": 0.2, "south": 0.8}},
    ],
}
