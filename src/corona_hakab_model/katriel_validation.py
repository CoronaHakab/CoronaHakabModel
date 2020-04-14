import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib_set_backend

from consts import Consts
from generation.circles_consts import CirclesConsts
from generation.connection_types import ConnectionTypes
from generation.generation_manager import GenerationManger
from generation.matrix_consts import MatrixConsts
from manager import SimulationManager
from medical_state import SusceptibleState, ContagiousState, ImmuneState
from medical_state_machine import MedicalStateMachine
from state_machine import TerminalState, StochasticState
from supervisor import Supervisable, LambdaValueSupervisable
from util import dist


def CalcBinomialInfected(num_of_days, r0, population_size, p_tau, initial_infected_count_num):
    """
    Calculates the average and the standard deviation of the number of infected individuals assuming a Binomial
    model. This should be more exact than the Poisson distribution, if the number of infected is very large (
    approaching the population size N). The infection probability calculated within corresponds to eq. (2.3) in the
    paper  "STOCHASTIC DISCRETE-TIME AGE-OF-INFECTION EPIDEMIC MODELS", G. Kartiel, International Journal of
    Biomathematics Vol. 6, No. 1 (2013) DOI: 10.1142/S1793524512500660 Input: num_of_days - duration of the
    simulation in days r0 - total number of people that a single individual infects during the whole period of his
    illness population size p_tau - a vector of length d where d is the number of illness days and p_tau[n] is the
    probability of infecting others on day n of the illness initial_infected_count_num

    Output:
    mean - vector of mean number of infected individuals along the num_of_days period
    sigma - vector of standard deviations of infected individuals along the num_of_days period
    """

    prev_s = population_size - initial_infected_count_num
    illness_duration = len(p_tau)
    prev_i = [0] * (illness_duration - 1) + [initial_infected_count_num]
    beta = r0 / np.sum(p_tau)  # number of people that a single sick individual infects during a single day (on average)

    mean = np.zeros(num_of_days)  # to store the number of (average) newly infected aside
    variance = np.zeros(num_of_days)  # to store the std of newly infected aside

    for current_day in range(num_of_days):
        # calc p of the Binomial distribution:
        p = 1 - np.exp(-(beta / population_size) * p_tau.dot(prev_i[::-1]))
        n = prev_s
        av_newInfections = n * p
        mean[current_day] = av_newInfections
        variance[current_day] = n * p * (1 - p)
        prev_i[0:-1] = prev_i[1:]  # shift old data
        prev_i[-1] = av_newInfections  # set the new infected as the last entry
        prev_s = prev_s - av_newInfections

    return mean, np.sqrt(variance)


class KatrielConsts(Consts):
    def __new__(cls, *args, r0, contagiousness, initial_infected_count, num_of_days, illness_duration, **kwargs):
        self = super().__new__(cls, *args,
                               r0=r0,
                               total_steps=num_of_days,
                               initial_infected_count=initial_infected_count,
                               **kwargs)
        self.illness_duration = illness_duration
        self.contagiousness = contagiousness
        return self

    def medical_state_machine(self) -> MedicalStateMachine:
        class SusceptibleTerminalState(SusceptibleState, TerminalState):
            pass

        class ContagiousStochasticState(ContagiousState, StochasticState):
            pass

        class ImmuneTerminalState(ImmuneState, TerminalState):
            pass

        susceptible = SusceptibleTerminalState("Susceptible", test_willingness=0)
        sick = ContagiousStochasticState(
            "Sick",
            contagiousness=self.contagiousness,
            test_willingness=0,
        )
        recovered = ImmuneTerminalState("Recovered", detectable=False, test_willingness=0)

        ret = MedicalStateMachine(susceptible, sick)
        sick.add_transfer(recovered, dist(self.illness_duration), ...)
        return ret


def run_sim():
    population_size = 3000
    r0 = 2.0
    initial_infected_count = 15
    num_of_days = 50
    illness_duration = 5
    contagiousness = 1 / illness_duration

    consts = KatrielConsts(
        r0=r0,
        num_of_days=num_of_days,
        initial_infected_count=initial_infected_count,
        contagiousness=contagiousness,
        illness_duration=illness_duration,
        partial_opening_active=False
    )

    cc = CirclesConsts(
        population_size=population_size,
        ages=[10, 40, 70],
        age_prob=[0.30, 0.45, 0.25],
        connection_type_prob_by_age_index=[
            {
                ConnectionTypes.Work: 0,
                ConnectionTypes.School: 0.0,
                ConnectionTypes.Family: 0.0,
                ConnectionTypes.Other: 1.0,
            },
            {
                ConnectionTypes.Work: 0,
                ConnectionTypes.School: 0,
                ConnectionTypes.Family: 0.0,
                ConnectionTypes.Other: 1.0,
            },
            {
                ConnectionTypes.Work: 0,
                ConnectionTypes.School: 0,
                ConnectionTypes.Family: 0.0,
                ConnectionTypes.Other: 1.0,
            },
        ],
        circle_size_distribution_by_connection_type={
            ConnectionTypes.School: ([100, 500, 1000, 1500], [0.03, 0.45, 0.35, 0.17]),
            ConnectionTypes.Work: ([1, 2, 10, 40, 300, 500], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
            ConnectionTypes.Family: ([100_000], [1]),
            ConnectionTypes.Other: ([100_000], [1.0]),
        },
        geo_circles_amount=1,
        geo_circles_names=["north"],
        geo_circles_agents_share=[1],
        multi_zone_connection_type_to_geo_circle_probability=[
            {ConnectionTypes.Work: {"North": 1}}
        ]

    )

    matrix_consts = MatrixConsts(
        connection_type_to_connection_strength={
            ConnectionTypes.Family: 3,
            ConnectionTypes.Work: 0.66,
            ConnectionTypes.School: 1,
            ConnectionTypes.Other: 0.23,
        },
        daily_connections_amount_by_connection_type={
            ConnectionTypes.School: 6,
            ConnectionTypes.Work: 5.6,
            ConnectionTypes.Other: 50.1,
        },
        weekly_connections_amount_by_connection_type={
            ConnectionTypes.School: 12.6,
            ConnectionTypes.Work: 12.6,
            ConnectionTypes.Other: 0,
        },
        community_triad_probability=(0,)
    )
    gm = GenerationManger(cc, matrix_consts)

    sm = SimulationManager(
        (
            Supervisable.NewCasesCounter(),
        ),
        gm.population_data,
        gm.matrix_data,
        consts=consts,
    )

    # Run binomial model
    p_tau = np.ones(illness_duration) * contagiousness
    mean, std = CalcBinomialInfected(num_of_days, r0, population_size, p_tau, initial_infected_count)

    # Run simulation

    print(sm)
    sm.run()
    df: pd.DataFrame = sm.dump(filename='katriel_test.csv')
    df.plot()

    plt.plot(np.arange(num_of_days), mean, 'm', linewidth=2, label='Binomial Model')
    plt.plot(np.arange(num_of_days), mean + std, '--m')
    plt.plot(np.arange(num_of_days), mean - std, '--m')
    plt.legend()
    plt.show()


run_sim()
