from itertools import islice
from time import time
from affinity_matrix import AffinityMatrix
import logging
import numpy as np
import plotting
import update_matrix
import random as rnd
import infection
from consts import Consts


class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    # GENERAL SIMULATION CONSTS:

    def __init__(self, consts=Consts()):
        self.consts = consts

        self.logger = logging.getLogger('simulation')
        logging.basicConfig()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating new simulation.")
        self.matrix = AffinityMatrix(self.consts.population_size, consts)
        self.agents = self.matrix.agents

        self.stats_plotter = plotting.StatisticsPlotter()
        self.update_matrix_manager = update_matrix.UpdateMatrixManager(self.matrix)
        self.infection_manager = infection.InfectionManager(self)

        self.step_counter = 0
        self.infected_per_generation = [0] * self.consts.total_steps
        self.recovered_per_generation = [0] * self.consts.total_steps
        self.dead_per_generation = [0] * self.consts.total_steps
        self.sick_per_generation = [0] * self.consts.total_steps
        self.recovered_counter = 0
        self.dead_counter = 0

        self.logger.info("Created new simulation.")

        # todo merge sick_agents and sick_agents_vector to one DS
        self.sick_agents = set()
        self.sick_agent_vector = np.zeros(self.consts.population_size, dtype=bool)

    def step(self):
        """
        run one step
        """
        # update matrix
        self.update_matrix_manager.update_matrix_step(self.infection_manager.agents_to_home_quarantine,
                                                      self.infection_manager.agents_to_full_quarantine)

        # update infection
        new_dead, new_recovered = \
            self.infection_manager.infection_step()

        # update stats
        self.dead_counter += new_dead
        self.recovered_counter += new_recovered
        self.update_stats()

        self.step_counter += 1

    def update_stats(self):
        self.recovered_per_generation[self.step_counter] = self.recovered_counter
        self.dead_per_generation[self.step_counter] = self.dead_counter
        self.sick_per_generation[self.step_counter] = len(self.sick_agents)
        self.infected_per_generation[self.step_counter] = len(
            self.sick_agents) + self.recovered_counter + self.dead_counter

    def setup_sick(self):
        """"
        setting up the simulation with a given amount of infected people
        """
        for agent in islice(self.agents, self.consts.initial_infected_count):
            agent.infect(0)
            self.sick_agents.add(agent)

    def generate_policy(self, workers_percent):
        """"
        setting up the simulation with a given amount of infected people
        """
        for agent in self.agents:
            if agent.work is None:
                continue
            if rnd.random() > workers_percent:
                work_members_ids = agent.work.get_indexes_of_my_circle(agent.ID)  # right now works are circle[1]
                for id in work_members_ids:
                    self.matrix.matrix[agent.ID, id] = np.log(1)
                family_members_ids = agent.home.get_indexes_of_my_circle(agent.ID)  # right now families are circle[0]
                for id in family_members_ids:
                    self.matrix.matrix[agent.ID, id] = \
                        np.log(1 - (self.consts.family_strength_not_workers*self.matrix.factor))
        self.setup_sick()

    def run(self):
        """
        runs full simulation
        """
        start_time = time()
        self.generate_policy(1)
        for i in range(self.consts.total_steps):
            if i == self.consts.stop_work_days:
                self.matrix.change_work_policy(False)
            elif i == self.consts.resume_work_days:
                self.matrix.change_work_policy(True)
            self.step()
            self.logger.info(
                f"performing step {i + 1}/{self.consts.total_steps} : "
                f"{self.sick_per_generation[i]} people are sick, "
                f"{self.recovered_per_generation[i]} people are recovered, "
                f"{self.dead_per_generation[i]} people are dead, "
                f"total amount of {self.infected_per_generation[i]} people were infected"
            )

    def plot(self):
        self.stats_plotter.plot_infected_per_generation(self.infected_per_generation, self.recovered_per_generation,
                                                        self.dead_per_generation, self.sick_per_generation)

    def __str__(self):
        return "<SimulationManager: SIZE_OF_POPULATION={}, STEPS_TO_RUN={}>".format(self.consts.population_size,
                                                                                    self.consts.total_steps)
