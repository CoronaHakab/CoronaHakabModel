import json
from datetime import datetime
import os
from collections import Counter
from typing import Dict, Tuple, List
import numpy as np

from common.agent import Agent
from consts import Consts
from generation.circles_consts import CirclesConsts
from common.medical_state import ImmuneState
from common.medical_state_machine import MedicalStateMachine
from medical_state_manager import MedicalStateManager
from project_structure import OUTPUT_FOLDER
from common.state_machine import TerminalState


def _generate_agents_randomly(population_size, ages_to_dist: dict) -> List:
    """
    Helper function for generating agents with ages for the state machine simulation
    :param population_size: Number of agents to produce
    :param circle_consts: CircleConsts - Used to generate agents ages
                                         according to circles configuration
    :return: List of agents
    """
    ages = np.random.choice(list(ages_to_dist.keys()),
                            size=population_size,
                            p=list(ages_to_dist.values()))
    return [Agent(index=uid, age=age) for (uid, age) in zip(range(population_size), ages)]


def _infect_all_agents(list_of_agents, medical_machine_manager, medical_state_machine):
    for agent in list_of_agents:
        agent.medical_state = medical_state_machine.default_state_upon_infection
    medical_state_machine.default_state_upon_infection.add_many(list_of_agents)

    # take list of agents and create a pending transfer from their initial state to the next state
    medical_machine_manager.pending_transfers.extend(
        medical_state_machine.default_state_upon_infection.transfer(list_of_agents)
    )


def _get_empirical_state_times(medical_state_machine: MedicalStateMachine,
                               population_size: int,
                               state_counter: Counter) -> Tuple[List]:
    """

    :param medical_state_machine: State machine representing the medical state
    :param population_size: The number of agents
    :param state_counter: Counter that represents total number of days
                          all the agents spent in the state overall
    :return: Two dictionaries:
             * The first one represents the empirical mean time spent in state,
               conditioned that we visit it
             * The second one represents the empirical mean time spent in state,
               taking in consideration the total population
    """
    average_state_time_duration = dict()
    state_duration_expected_time = dict()
    for m in medical_state_machine.states:
        if len(m.ever_visited) == 0:
            average_state_time_duration[m.name] = 0
            state_duration_expected_time[m.name] = 0
        else:
            average_state_time_duration[m.name] = state_counter[m.name] / len(m.ever_visited)
            state_duration_expected_time[m.name] = state_counter[m.name] / population_size
    return average_state_time_duration, state_duration_expected_time


def _is_terminal_state(state):
    """

    :param state: A state of the medical state machine
    :return: Whether this agent reached a state from which he will not infect anymore.
             In our scenario it means he died or recovered.
    """
    return isinstance(state, TerminalState) and isinstance(state, ImmuneState)


def monte_carlo_state_machine_analysis(configuration: Dict) -> Dict:
    """

    :param configuration: Dictionary with configuration for the mc run
           configuration must contain:
           * population_size for the mc
           configuration might contain:
           * consts_file - For loading Consts()
           * circle_consts file - for loading CircleConsts
    :return: Dictionary with statistics of the run:
                * population_size
                * days_passed - time it took to all the agents to recover/die
                * time_in_each_state - for each state, total number of days all agents were in it
                * visitors_in_each_state - Number of agents that visited each state
                * average_duration_in_state - Empirical mean time to stay
                                              at state conditioned that we enter it
                * state_duration_expected_time - Empirical mean to stay in state, not conditioned
                * average_time_to_terminal -Empirical mean time until death/recovery

    """
    if 'consts_file' in configuration:
        consts = Consts.from_file(configuration['consts_file'])
    else:
        consts = Consts()

    if "circle_consts_file" in configuration:
        circles_consts = CirclesConsts.from_file(configuration['circle_consts_file'])
        population_size = int(circles_consts.population_size)
    else:
        population_size = int(configuration["population_size"])

    assert "age_distribution" in configuration, "Must supply age distribution"
    ages_to_dist = configuration['age_distribution']
    medical_state_machine = consts.medical_state_machine()
    medical_machine_manager = MedicalStateManager(medical_state_machine=medical_state_machine)
    agents_list = _generate_agents_randomly(population_size=population_size,
                                            ages_to_dist=ages_to_dist)
    _infect_all_agents(agents_list, medical_machine_manager, medical_state_machine)
    medical_states = medical_state_machine.states
    terminal_states = list(filter(_is_terminal_state,
                                  medical_states))

    state_counter = Counter({m.name: m.agent_count for m in medical_states})
    sum_days_to_terminal = 0
    days_passed = 1
    number_terminals_agents = 0

    while number_terminals_agents != population_size:
        # No manager so we don't update it
        previous_terminal_agents = sum([m.agent_count for m in terminal_states])
        medical_machine_manager.step(list())
        number_terminals_agents = sum([m.agent_count for m in terminal_states])
        new_terminals = number_terminals_agents - previous_terminal_agents
        for m in medical_states:
            state_counter[m.name] += m.agent_count
        days_passed += 1
        sum_days_to_terminal += days_passed * new_terminals

    average_state_time_duration, state_duration_expected_time = _get_empirical_state_times(medical_state_machine,
                                                                                           population_size,
                                                                                           state_counter)

    return dict(population_size=population_size,
                days_passed=days_passed,
                time_in_each_state=dict(state_counter),
                visitors_in_each_state={m.name: len(m.ever_visited)
                                        for m in medical_state_machine.states},
                average_duration_in_state=average_state_time_duration,
                state_duration_expected_time=state_duration_expected_time,
                average_time_to_terminal=sum_days_to_terminal/population_size)


def extract_state_machine_analysis(configuration):
    output_file = OUTPUT_FOLDER /\
                  f"state_machine_analysis_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    result = monte_carlo_state_machine_analysis(configuration)
    with open(output_file, 'w') as result_json:
        json.dump(result, result_json, indent=4, sort_keys=True)


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame()
    for i in [1000, 5000, 10000, 25_000, 50_000, 100_000]:
        monte_carlo_config = dict(population_size=i,
                                  age_distribution={8: 1})
        result = monte_carlo_state_machine_analysis(monte_carlo_config)
        dataframe_dict = dict()
        for k, v in result.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    dataframe_dict[k+"_"+k2] = v2
            else:
                dataframe_dict[k] = v
        df = df.append(dataframe_dict, ignore_index=True)
        print(f"Finished population of size {i}")
    file_name = os.path.join(OUTPUT_FOLDER,
                             f"simulation_statistics_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    df = df.set_index("population_size")
    df.to_csv(file_name)
