import logging
import random as rnd
from itertools import islice
from time import time

import infection
import numpy as np
import plotting
import update_matrix
from affinity_matrix import AffinityMatrix
from consts import Consts
from medical_state import MedicalState


class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    # GENERAL SIMULATION CONSTS:

    def __init__(self, consts=Consts()):
        self.consts = consts

        self.logger = logging.getLogger("simulation")
        logging.basicConfig()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating new simulation.")
        self.matrix = AffinityMatrix(self.consts.population_size, consts)
        self.agents = self.matrix.agents

        self.stats_plotter = plotting.StatisticsPlotter()
        self.update_matrix_manager = update_matrix.UpdateMatrixManager(self.matrix)
        self.infection_manager = infection.InfectionManager(self)

        self.step_counter = 0

        self.per_generation = {}
        self.counters = {}
        for state in MedicalState:
            self.per_generation[state] = [0] * self.consts.total_steps
            self.counters[state] = 0
        self.per_generation["total infected"] = [0] * self.consts.total_steps
        self.per_generation["total current sick"] = [0] * self.consts.total_steps

        self.logger.info("Created new simulation.")

        # todo merge sick_agents and sick_agents_vector to one DS
        self.sick_agents = set()
        self.sick_agent_vector = np.zeros(self.consts.population_size, dtype=bool)

    def step(self):
        """
        run one step
        """
        # update matrix
        self.update_matrix_manager.update_matrix_step(
            self.infection_manager.agents_to_home_quarantine,
            self.infection_manager.agents_to_full_quarantine,
        )
        # update infection
        self.infection_manager.infection_step()
        # update stats
        self.update_stats()

        self.step_counter += 1

    def update_stats(self):
        for state in MedicalState:
            self.per_generation[state][self.step_counter] = self.counters[state]
        self.per_generation["total current sick"][self.step_counter] = (
            sum(self.counters.values())
            - self.counters[MedicalState.Deceased]
            - self.counters[MedicalState.Immune]
        )
        self.per_generation["total infected"][self.step_counter] = sum(
            self.counters.values()
        )

    def setup_sick(self):
        """"
        setting up the simulation with a given amount of infected people
        """
        for agent in islice(self.agents, self.consts.initial_infected_count):
            agent.infect(0, self)
            self.sick_agents.add(agent)

    def generate_policy(self, workers_percent):
        """"
        setting up the simulation with a given amount of infected people
        """
        for agent in self.agents:
            if agent.work is None:
                continue
            if rnd.random() > workers_percent:
                work_members_ids = agent.work.get_indexes_of_my_circle(
                    agent.ID
                )  # right now works are circle[1]
                for id in work_members_ids:
                    self.matrix.matrix[agent.ID, id] = np.log(1)
                family_members_ids = agent.home.get_indexes_of_my_circle(
                    agent.ID
                )  # right now families are circle[0]
                for id in family_members_ids:
                    self.matrix.matrix[agent.ID, id] = np.log(
                        1
                        - (self.consts.family_strength_not_workers * self.matrix.factor)
                    )
        self.setup_sick()

    def run(self):
        """
        runs full simulation
        """
        start_time = time()
        self.generate_policy(1)
        for i in range(self.consts.total_steps):
            if Consts.active_quarantine:
                if i == self.consts.stop_work_days:
                    self.matrix.change_work_policy(False)
                elif i == self.consts.resume_work_days:
                    self.matrix.change_work_policy(True)
            self.step()
            self.logger.info(
                "performing step {}/{} :{} people are sick, {} people are recovered, {} people are dead, total amount of {} people were infected".format(
                    i + 1,
                    self.consts.total_steps,
                    self.per_generation["total current sick"][i],
                    self.per_generation[MedicalState.Immune][i],
                    self.per_generation[MedicalState.Deceased][i],
                    self.per_generation["total infected"][i],
                )
            )
        runtime = time() - start_time
        print(f"--- {runtime} seconds ---")

    def plot(self):
        self.stats_plotter.plot_infected_per_generation(self.per_generation)

    def __str__(self):
        return "<SimulationManager: SIZE_OF_POPULATION={}, STEPS_TO_RUN={}>".format(
            self.consts.population_size, self.consts.total_steps
        )
