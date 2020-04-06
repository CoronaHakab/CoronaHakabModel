import matplotlib.pyplot as plt
import numpy as np

from consts import Consts
from generation.generation_manager import GenerationManger
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

    def __init__(self, r0, contagiousness, initial_infected_count, num_of_days, illness_duration) -> None:
        super().__init__(
            r0=r0,
            total_steps=num_of_days,
            initial_infected_count=initial_infected_count,
            active_isolation=False
        )
        self.illness_duration = illness_duration
        self.contagiousness = contagiousness

    def medical_state_machine(self) -> MedicalStateMachine:
        class SusceptibleTerminalState(SusceptibleState, TerminalState):
            pass

        class ContagiousStochasticState(ContagiousState, StochasticState):
            pass

        class ImmuneTerminalState(ImmuneState, TerminalState):
            pass

        susceptible = SusceptibleTerminalState("Susceptible")
        sick = ContagiousStochasticState(
            "Sick",
            contagiousness=self.contagiousness,
            test_willingness=1,
        )
        recovered = ImmuneTerminalState("Recovered", detectable=False)

        ret = MedicalStateMachine(susceptible, sick)
        sick.add_transfer(recovered, dist(self.illness_duration), ...)
        return ret


def run_sim():
    r0 = 2.0
    contagiousness = 0.2
    initial_infected_count = 15
    num_of_days = 150
    illness_duration = 5

    consts = KatrielConsts(r0, contagiousness, initial_infected_count, num_of_days, illness_duration)

    gm = GenerationManger()
    population_size = len(gm.population_data.agents)
    sm = SimulationManager(
        (
            Supervisable.NewCasesCounter(),
        ),
        gm.population_data,
        gm.matrix_data,
        consts=consts,
    )
    print(sm)
    sm.run()
    sm.plot(save=True, max_scale=False)

run_sim()