import logging
from collections import defaultdict
from typing import Any, Iterable

import infection
import numpy as np
import update_matrix
from affinity_matrix import AffinityMatrix
from agent import Agent
from consts import Consts
from state_machine import PendingTransfers
from supervisor import Supervisable, Supervisor


class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    def __init__(
        self,
        supervisable_makers: Iterable[Any],
        consts=Consts(),
        input_matrix_path: str = None,
        output_matrix_path: str = None,
    ):
        self.consts = consts
        self.medical_machine = consts.medical_state_machine()
        initial_state = self.medical_machine.initial

        self.pending_transfers = PendingTransfers()

        self.logger = logging.getLogger("simulation")
        logging.basicConfig()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating a new simulation.")
        self.logger.info(f"Generating {self.consts.population_size} agents")

        # the manager holds the vector, but the agents update it
        self.contagiousness_vector = np.empty(self.consts.population_size, dtype=float)  # how likely to infect others
        self.susceptible_vector = np.empty(self.consts.population_size, dtype=bool)  # can get infected
        self.agents = [Agent(i, self, initial_state) for i in range(self.consts.population_size)]
        initial_state.add_many(self.agents)

        self.matrix = AffinityMatrix(self, input_matrix_path, output_matrix_path)

        self.supervisor = Supervisor([Supervisable.coerce(a, self) for a in supervisable_makers], self)
        self.update_matrix_manager = update_matrix.UpdateMatrixManager(self.matrix)
        self.infection_manager = infection.InfectionManager(self)

        self.current_date = 0

        self.logger.info("Created new simulation.")

    def step(self):
        """
        run one step
        """
        # update matrix
        self.update_matrix_manager.update_matrix_step(
            self.infection_manager.agents_to_home_isolation, self.infection_manager.agents_to_full_isolation,
        )

        # run infection
        new_sick = self.infection_manager.infection_step()

        # progress transfers
        self.progress_transfers(new_sick)

        self.current_date += 1

        self.supervisor.snapshot(self)

    def progress_transfers(self, new_sick):
        # all the new sick agents are leaving their previous step
        changed_state_leaving = new_sick

        # agents which are going to enter the new state
        changed_state_introduced = defaultdict(list)
        # list of all the new sick agents
        new_sick_list = sum(changed_state_leaving.values(), [])
        # all the new sick are going to get to the next state
        changed_state_introduced[self.medical_machine.state_upon_infection] = new_sick_list

        for s in new_sick_list:
            s.set_medical_state_no_inform(self.medical_machine.state_upon_infection)

        moved = self.pending_transfers.advance()
        for (agent, destination, origin, _) in moved:
            agent.set_medical_state_no_inform(destination)

            changed_state_introduced[destination].append(agent)
            changed_state_leaving[origin].append(agent)

        for state, agents in changed_state_introduced.items():
            state.add_many(agents)
            self.pending_transfers.extend(state.transfer(agents))

        for state, agents in changed_state_leaving.items():
            state.remove_many(agents)

    def setup_sick(self):
        """"
        setting up the simulation with a given amount of infected people
        """
        # todo we only do this once so it's fine but we should really do something better
        agents_to_infect = self.agents[: self.consts.initial_infected_count]

        for agent in agents_to_infect:
            agent.set_medical_state_no_inform(self.medical_machine.state_upon_infection)

        self.medical_machine.initial.remove_many(agents_to_infect)
        self.medical_machine.state_upon_infection.add_many(agents_to_infect)

        # take list of agents and create a pending transfer from their initial state to the next state
        self.pending_transfers.extend(self.medical_machine.state_upon_infection.transfer(agents_to_infect))

    def run(self):
        """
        runs full simulation
        """
        self.setup_sick()

        for i in range(self.consts.total_steps):
            if Consts.active_isolation:
                if i == self.consts.stop_work_days:
                    self.matrix.change_connections_policy({"home", "strangers"})
                elif i == self.consts.resume_work_days:
                    self.matrix.change_connections_policy({"home", "strangers", "school", "work"})
            self.step()
            self.logger.info(f"performing step {i + 1}/{self.consts.total_steps}")

    def plot(self, **kwargs):
        self.supervisor.plot(**kwargs)

    def stackplot(self, **kwargs):
        self.supervisor.stack_plot(**kwargs)

    def __str__(self):
        return "<SimulationManager: SIZE_OF_POPULATION={}, STEPS_TO_RUN={}>".format(
            self.consts.population_size, self.consts.total_steps
        )
