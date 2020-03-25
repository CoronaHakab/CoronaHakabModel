from collections import defaultdict
from typing import Tuple, Any, Iterable

from affinity_matrix import AffinityMatrix
import logging
import numpy as np
import update_matrix
import infection
from agent import Agent
from consts import Consts
from state_machine import PendingTransfers
from supervisor import Supervisor, Supervisable


class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    def __init__(self, supervisable_makers: Iterable[Any], consts=Consts()):
        self.consts = consts
        self.medical_machine = consts.medical_state_machine()
        initial_state = self.medical_machine.initial

        self.pending_transfers = PendingTransfers()

        self.logger = logging.getLogger('simulation')
        logging.basicConfig()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating new simulation.")
        self.logger.info(f"Generating {self.consts.population_size} agents")

        # the manager holds the vector, but the agents update it
        self.infectiousness_vector = np.empty(self.consts.population_size, dtype=float)
        self.infectable_vector = np.empty(self.consts.population_size, dtype=bool)
        self.agents = [Agent(i, self, initial_state) for i in range(self.consts.population_size)]
        initial_state.add_many(self.agents)

        self.matrix = AffinityMatrix(self)

        self.supervisor = Supervisor(
            [Supervisable.coerce(a)(self) for a in supervisable_makers], self
        )
        self.update_matrix_manager = update_matrix.UpdateMatrixManager(self.matrix)
        self.infection_manager = infection.InfectionManager(self)

        self.current_date = 0

        self.logger.info("Created new simulation.")

    def step(self):
        """
        run one step
        """
        # update matrix
        self.update_matrix_manager.update_matrix_step(self.infection_manager.agents_to_home_quarantine,
                                                      self.infection_manager.agents_to_full_quarantine)

        # run infection
        new_sick = self.infection_manager.infection_step()

        # progress transfers
        self.progress_transfers(new_sick)

        self.current_date += 1

        self.supervisor.snapshot(self)

    def progress_transfers(self, new_sick):
        changed_state_introduced = defaultdict(list)
        changed_state_leaving = new_sick

        all_sick = sum(
            changed_state_leaving.values(), []
        )

        changed_state_introduced[self.medical_machine.state_upon_infection] = all_sick

        for s in all_sick:
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
        agents_to_infect = self.agents[:self.consts.initial_infected_count]

        for agent in agents_to_infect:
            agent.set_medical_state_no_inform(self.medical_machine.state_upon_infection)

        self.medical_machine.initial.remove_many(agents_to_infect)
        self.medical_machine.state_upon_infection.add_many(agents_to_infect)

        self.pending_transfers.extend(
            self.medical_machine.state_upon_infection.transfer(agents_to_infect)
        )

    def generate_policy(self, workers_percent):
        """"
        setting up the simulation with a given amount of infected people
        """
        rolls = np.random.random(len(self.agents)) > workers_percent
        for agent, roll in zip(self.agents, rolls):
            if agent.work is None:
                continue
            if roll:
                work_members_ids = agent.work.get_indexes_of_my_circle(agent.index)  # right now works are circle[1]
                for id in work_members_ids:
                    self.matrix.matrix[agent.index, id] = np.log(1)
                family_members_ids = agent.home.get_indexes_of_my_circle(
                    agent.index)  # right now families are circle[0]
                for id in family_members_ids:
                    self.matrix.matrix[agent.index, id] = \
                        np.log(1 - (self.consts.family_strength_not_workers * self.matrix.factor))
        self.setup_sick()

    def run(self):
        """
        runs full simulation
        """
        self.generate_policy(1)

        for i in range(self.consts.total_steps):
            if Consts.active_quarantine:
                if i == self.consts.stop_work_days:
                    self.matrix.change_work_policy(False)
                elif i == self.consts.resume_work_days:
                    self.matrix.change_work_policy(True)
            self.step()
            self.logger.info(
                f"performing step {i + 1}/{self.consts.total_steps}"
            )

    def plot(self, **kwargs):
        self.supervisor.plot(**kwargs)

    def __str__(self):
        return "<SimulationManager: SIZE_OF_POPULATION={}, STEPS_TO_RUN={}>".format(self.consts.population_size,
                                                                                    self.consts.total_steps)
