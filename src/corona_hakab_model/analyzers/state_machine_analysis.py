from datetime import datetime
import json
import os
from collections import Counter
from agent import Agent
from consts import Consts
from generation.circles_consts import CirclesConsts
from medical_state_manager import MedicalStateManager
from state_machine import TerminalState


def _generate_agents_randomly(population_size, circle_consts):
    list_of_agents = []
    age_dist = circle_consts.get_age_distribution()
    for i in range(population_size):
        new_agent = Agent(index=i)
        new_agent.age = age_dist.rvs()
        list_of_agents.append(new_agent)
    return list_of_agents


def _infect_all_agents(list_of_agents, medical_machine_manager, medical_state_machine):
    for agent in list_of_agents:
        agent.medical_state = medical_state_machine.default_state_upon_infection
    medical_state_machine.default_state_upon_infection.add_many(list_of_agents)

    # take list of agents and create a pending transfer from their initial state to the next state
    medical_machine_manager.pending_transfers.extend(
        medical_state_machine.default_state_upon_infection.transfer(list_of_agents)
    )


def run_monte_carlo(configuration):

    if 'consts_file' in configuration:
        consts = Consts.from_file(configuration['consts_file'])
    else:
        consts = Consts()

    if "circle_consts_file" in configuration:
        consts = CirclesConsts.from_file(configuration['circle_consts_file'])
    else:
        circle_const = CirclesConsts()

    population_size = configuration["monte_carlo_size"]

    medical_state_machine = consts.medical_state_machine()
    medical_machine_manager = MedicalStateManager(medical_state_machine)
    agents_list = _generate_agents_randomly(population_size=population_size, circle_consts=circle_const)
    _infect_all_agents(agents_list, medical_machine_manager, medical_state_machine)

    days_passed = 1
    state_counter = Counter()
    state_visitors = dict()
    for agent in agents_list:
        state_name = agent.medical_state.name
        agent_id = agent.index
        if state_name in state_visitors:
            state_visitors[state_name].add(agent_id)
        else:
            state_visitors[state_name] = set([agent_id])
        state_counter[state_name] += 1

    done = False
    while not done:
        # Calculate the new states
        # Everybody is infected so no new infected
        # No manager so we don't update it
        medical_machine_manager.step(list(), update_only_medical_state=True)

        # Since they all start with infected, non are susceptible
        not_terminal = filter(lambda current_agent: not isinstance(current_agent.medical_state, TerminalState),
                              agents_list)
        not_terminal = list(not_terminal)

        for sick_agent in not_terminal:
            current_state_name = sick_agent.medical_state.name
            state_counter[current_state_name] += 1
            if current_state_name in state_visitors:
                state_visitors[current_state_name].add(sick_agent.index)
            else:
                state_visitors[current_state_name] = set([sick_agent.index])

        days_passed += 1
        done = len(not_terminal) == 0
    state_visitors_count = dict([(state, len(visitors))  for state, visitors in state_visitors.items()])
    average_state_time_duration = dict([(k, state_counter[k]/state_visitors_count[k])
                                        for k in state_counter])
    return dict(population_size=population_size,
                days_passed=days_passed,
                time_in_each_state=dict(state_counter),
                visitors_in_each_state=dict(state_visitors_count),
                average_duration_in_state=average_state_time_duration)


if __name__ == "__main__":
    monte_carlo_config = dict(monte_carlo_size=1_000_000)
    result = run_monte_carlo(monte_carlo_config)
    output_folder = "../../output" # There is a constant, but come on...
    file_name = os.path.join(output_folder,
                             f"simulation_statistics_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(file_name, 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=4)
