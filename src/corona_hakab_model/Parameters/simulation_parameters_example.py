{
    "total_steps": 350,
    "initial_infected_count": 20,
    "export_infected_agents_interval": 50,

    # Size of population to estimate expected time for each state
    "population_size_for_state_machine_analysis": 25_000,

    # Tsvika: Currently the distribution is selected based on the number of input parameters.
    # Think we should do something more readable later on.
    # For example: "latent_to_silent_days": {"type":"uniform","lower_bound":1,"upper_bound":3}
    # disease states transition lengths distributions
    "latent_to_pre_symptomatic_days": dist(1, 5, 10),
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10],
    # [0.022,0.052,0.082,0.158,0.234,0.158,0.152,0.082,0.04,0.02]))
    "latent_to_asymptomatic_days": dist(1, 5, 11),
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11],
    # [0.02,0.05,0.08,0.15,0.22,0.15,0.15,0.08,0.05,0.03,0.02]))
    "pre_symptomatic_to_mild_condition_days": dist(1, 3),
    "mild_to_close_medical_care_days": dist(3, 11),
    # Actual distribution: rv_discrete(values=([3,4,5,6,7,8,9,10,11,12],
    # [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.01]))
    "mild_to_need_icu_days": dist(6, 13, 29),
    # Actual distribution: rv_discrete(values=([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    # [0.012,0.019,0.032,0.046,0.059,0.069,0.076,0.078,0.076,0.072,0.066,0.060,0.053,0.046,0.040,0.035,0.030,0.028,0.026,0.022,0.020,0.015,0.010,0.010]))
    "mild_to_pre_recovered_days": dist(1, 17, 28),
    # Actual distribution: rv_discrete(values=(
    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],
    # [0.001,0.001,0.001,0.001,0.001,0.002,0.004,0.008,0.013,0.022,0.032,0.046,0.06,0.075,0.088,0.097,0.1,0.098,0.088,0.075,0.06,0.046,0.032,0.022,0.013,0.008,0.004,0.002]))
    "close_medical_care_to_icu_days": dist(10, 12, 14),
    "close_medical_care_to_mild_days": dist(8, 10, 12),
    "need_icu_to_deceased_days": dist(1, 3, 20),
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    # [0.030,0.102,0.126,0.112,0.090,0.080,0.075,0.070,0.065,0.050,0.040,0.035,0.030,0.025,0.020,
    # 0.015,0.012,0.010,0.008,0.005]))
    "need_icu_to_improving_days": dist(1, 5, 25),
    # Actual distribution: rv_discrete(values=([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
    # [0.021,0.041,0.081,0.101,0.101,0.081,0.071,0.066,0.061,0.056,0.046,0.041,0.039,0.033,0.031,0.026,0.021,0.016,0.013,0.013,0.011,0.011,0.009,0.005,0.005]))
    "improving_to_need_icu_days": dist(21, 42),
    "improving_to_pre_recovered_days": dist(21, 42),
    "improving_to_mild_condition_days": dist(21, 42),
    "pre_recovered_to_recovered_days": dist(14, 28),
    # Actual distribution: rv_discrete(values=([14, 28], [0.8, 0.2]))
    "asymptomatic_to_recovered_days": dist(10, 18, 35),
    # Actual distribution: rv_discrete(values=(
    # [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    # [0.013,0.016,0.025,0.035,0.045,0.053,0.061,0.065,0.069,0.069,0.065,0.063,0.058,0.053,0.056,0.041,0.040,0.033,
    # 0.030,0.025,0.020,0.015,0.015,0.015,0.010,0.010]))
    # infections ratios, See bucket dict for more info on how to use.
    "pre_symptomatic_infection_ratio": BucketDict({10: 0.75, 20: 0.75}), # x <= 10 then key is 10,
    "mild_condition_infection_ratio": BucketDict({10: 0.40}), # x<=20 then key is 20,
    "silent_infection_ratio": BucketDict({10: 0.3}), # if x greater than biggest key, x is biggest key
    # base r0 of the disease
    "r0": 2.4,

    # --Detection tests params--
    # the probability that an infected agent is asking to be tested
    "susceptible_test_willingness": 0.01,
    "latent_test_willingness": 0.01,
    "asymptomatic_test_willingness": 0.01,
    "pre_symptomatic_test_willingness": 0.01,
    "mild_condition_test_willingness": 0.6,
    "need_close_medical_care_test_willingness": 0.9,
    "need_icu_test_willingness": 1.0,
    "improving_health_test_willingness": 1.0,
    "pre_recovered_test_willingness": 0.5,
    "recovered_test_willingness": 0.1,
    "detection_pool": [
    DetectionSettings(
        name="hospital",
        detection_test=DetectionTest(detection_prob=0.98,
                                     false_alarm_prob=0.,
                                     time_until_result=3),
        daily_num_of_tests_schedule={0: 100, 10: 1000, 20: 2000, 50: 5000},
        testing_gap_after_positive_test=2,
        testing_gap_after_negative_test=1,
        testing_priorities=[
            DetectionPriority(
                lambda agent: (agent.medical_state.name == "Symptomatic" and
                               agent not in agent.manager.tested_positive_vector),
                max_tests=100),
            DetectionPriority(
                lambda agent: agent.medical_state.name == "Recovered"),
        ]),

    DetectionSettings(
        name="street",
        detection_test=DetectionTest(detection_prob=0.92,
                                     false_alarm_prob=0.,
                                     time_until_result=5),
        daily_num_of_tests_schedule={0: 500, 10: 1500, 20: 2500, 50: 7000},
        testing_gap_after_positive_test=3,
        testing_gap_after_negative_test=1,
        testing_priorities=[
            DetectionPriority(
                lambda agent: agent.medical_state.name == "Symptomatic"),
            DetectionPriority(
                lambda agent: agent.medical_state.name == "Recovered"),
        ]),
    ],
    "should_isolate_positive_detected": False,
    # --policies params--
    "change_policies": False,
    # a dictionary of day:([ConnectionTypes], message). on each day, keeps only the given connection types opened
    "policies_changes": {
        40: ([ConnectionTypes.Family, ConnectionTypes.Other], "closing schools, kindergartens and works"),
        70: ([ConnectionTypes.Family, ConnectionTypes.Other, ConnectionTypes.School, ConnectionTypes.Kindergarten],
             "opening schools and kindergartens"),
        100: (ConnectionTypes, "opening works"),
    },
    # policies acting on a specific connection type, when a term is satisfied
    "partial_opening_active": True,
    # each connection type gets a list of conditioned policies.
    # each conditioned policy actives a specific policy when a condition is satisfied.
    # each policy changes the multiplication factor of a specific circle.
    # each policy is activated only if a list of terms is fulfilled.
    "connection_type_to_conditioned_policy": {
    ConnectionTypes.School: [
        ConditionedPolicy(
            activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
            policy=Policy(0, [lambda circle: random() > 0]),
            message="closing all schools",
        ),
        ConditionedPolicy(
            activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
            policy=Policy(1, [lambda circle: random() > 1]),
            active=True,
            message="opening all schools",
        ),
    ],
    ConnectionTypes.Kindergarten: [
        ConditionedPolicy(
            activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
            policy=Policy(0, [lambda circle: random() > 0]),
            message="closing all kindergartens",
        ),
        ConditionedPolicy(
            activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
            policy=Policy(1, [lambda circle: random() > 1]),
            active=True,
            message="opening all kindergartens",
        ),
    ],
    ConnectionTypes.Work: [
        ConditionedPolicy(
            activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) > 1000,
            policy=Policy(0, [lambda circle: random() > 0]),
            message="closing all workplaces",
        ),
        ConditionedPolicy(
            activating_condition=lambda kwargs: len(np.flatnonzero(kwargs["manager"].contagiousness_vector)) < 500,
            policy=Policy(0, [lambda circle: random() > 1]),
            active=True,
            message="opening all workplaces",
        ),
    ],
    },
}