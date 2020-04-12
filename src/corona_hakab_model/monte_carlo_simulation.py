from collections import Counter
from agent import Agent
from consts import Consts
from medical_state_manager import MedicalStateManager
from state_machine import TerminalState


def generate_agents_randomly(population_size):
    list_of_agents = []
    for i in range(population_size):
        list_of_agents.append(Agent(index=i))
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

    population_size = configuration["monte_carlo_size"]

    medical_state_machine = consts.medical_state_machine()
    medical_machine_manager = MedicalStateManager(medical_state_machine)
    agents_list = generate_agents_randomly(population_size=population_size)
    _infect_all_agents(agents_list, medical_machine_manager, medical_state_machine)

    days_passed = 0
    state_counter = Counter()
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

        for agent in not_terminal:
            state_counter[agent.medical_state.name] += 1

        days_passed += 1
        done = len(not_terminal) == 0

    print(f"{state_counter=}")
    print(f"Simulation ended after {days_passed} days")


if __name__ == "__main__":
    monte_carlo_config = dict(monte_carlo_size=10_000)
    run_monte_carlo(monte_carlo_config)
