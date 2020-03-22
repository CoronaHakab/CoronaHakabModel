from itertools import islice
from time import time

from affinity_matrix import AffinityMAtrix
import logging
import numpy as np
import plotting
import update_matrix


class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    # GENERAL SIMULATION CONSTS:
    SIZE_OF_POPULATION = 10_000
    STEPS_TO_RUN = 150

    def __init__(self):
        self.matrix = AffinityMAtrix(self.SIZE_OF_POPULATION)
        self.agents = self.matrix.agents
        logging.basicConfig()
        self.logger = logging.getLogger('simulation')
        self.logger.setLevel(logging.INFO)
        self.stats_plotter = plotting.StatisticsPlotter()
        self.update_matrix_manager = update_matrix.UpdateMatrixManager()

        self.step_counter = 0
        self.infected_per_generation = [0] * self.STEPS_TO_RUN
        self.recovered_per_generation = [0] * self.STEPS_TO_RUN
        self.dead_per_generation = [0] * self.STEPS_TO_RUN
        self.sick_per_generation = [0] * self.STEPS_TO_RUN
        self.recovered_counter = 0
        self.dead_counter = 0

        self.logger.info("Created new simulation.")

        # todo merge sick_agents and sick_agents_vector to one DS
        self.sick_agents = set()
        self.sick_agent_vector = np.zeros(self.SIZE_OF_POPULATION, dtype=bool)

    def _update_matrix(self):
        """
        update matrix using current policies
        """
        self._update_sick_agents()
        # self.update_matrix_manager.update_matrix_step()

    def _update_sick_agents(self):
        """
        after each day, go through all sick agents and updates their status (allowing them to recover or die)
        """
        to_remove = set()
        rolls = np.random.random(len(self.sick_agents))
        for agent, roll in zip(self.sick_agents, rolls):
            result = agent.day_passed(roll, self.step_counter)
            if result:
                to_remove.add(agent)
                self.sick_agent_vector[agent.ID] = False
                if result == "Dead":
                    self.dead_counter = self.dead_counter + 1
                elif result == "Recovered":
                    self.recovered_counter = self.recovered_counter + 1
        self.sick_agents.difference_update(to_remove)

    def _perform_infection(self):
        """
        perform the infection stage by multiply matrix with infected vector and try to infect agents.

        v = [i for i in self.agents.is_infectious]
        perform w*v
        for each person in v:
            if rand() < v[i]
                agents[i].infect
        """

        v = self.sick_agent_vector

        u = self.matrix.matrix.dot(v)
        infections = np.random.random(u.shape) < (1 - np.exp(u))
        for agent, value in zip(self.agents, infections):
            if value and agent.infect(self.step_counter):
                self.sick_agent_vector[agent.ID] = True
                self.sick_agents.add(agent)

    def step(self):
        """
        run one step
        """

        self._update_matrix()
        self._perform_infection()
        self.update_stats()
        self.step_counter += 1

    def update_stats(self):
        self.recovered_per_generation[self.step_counter] = self.recovered_counter
        self.dead_per_generation[self.step_counter] = self.dead_counter
        self.sick_per_generation[self.step_counter] = len(self.sick_agents)
        self.infected_per_generation[self.step_counter] = len(
            self.sick_agents) + self.recovered_counter + self.dead_counter

    def setup_sick(self, amount_of_infected_to_start_with):
        """"
        setting up the simulation with a given amount of infected people
        """
        for agent in islice(self.agents, amount_of_infected_to_start_with):
            agent.infect(0)
            self.sick_agents.add(agent)

        self.sick_agent_vector[:amount_of_infected_to_start_with] = True

    def run(self):
        """
        runs full simulation
        """
        self.setup_sick(5)
        start_time = time()
        for i in range(self.STEPS_TO_RUN):
            self.step()
            self.logger.info(
                "performing step {}/{} : {} people are sick, {} people are recovered, {} people are dead, total amount of {} people were infected".format(
                    i,
                    self.STEPS_TO_RUN,
                    self.sick_per_generation[
                        i],
                    self.recovered_per_generation[
                        i],
                    self.dead_per_generation[
                        i], self.infected_per_generation[i]))

        # plot results
        # logoritmic scale:
        # self.stats_plotter.plot_infected_per_generation(list(map(lambda o: np.log(o), self.infected_per_generation)))
        # linear scale:
        # self.stats_plotter.plot_log_with_linear_regression(self.sick_per_generation)

    def plot(self):
        self.stats_plotter.plot_infected_per_generation(self.sick_per_generation)

    def __str__(self):
        return "<SimulationManager: SIZE_OF_POPULATION={}, STEPS_TO_RUN={}>".format(self.SIZE_OF_POPULATION,
                                                                                    self.STEPS_TO_RUN)
