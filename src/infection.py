import numpy as np

from medical_state import MedicalState

import manager


class InfectionManager:
    """
    Manages the infection stage
    """

    def __init__(self, sim_manager: 'manager.SimulationManager'):
        self.agents_to_home_quarantine = []
        self.agents_to_full_quarantine = []
        self.manager = sim_manager

    def infection_step(self):
        # perform infection
        self.agents_to_home_quarantine.clear()
        self.agents_to_full_quarantine.clear()
        self._perform_infection()

        # update agents
        new_dead, new_recovered = self._update_sick_agents()

        return new_dead, new_recovered

    def _perform_infection(self):
        """
        perform the infection stage by multiply matrix with infected vector and try to infect agents.

        v = [i for i in self.agents.is_infectious]
        perform w*v
        for each person in v:
            if rand() < v[i]
                agents[i].infect
        """

        v = np.copy(self.manager.sick_agent_vector)

        # a patch just for the mvp, in order to simulate infection ratio
        rolls = np.random.random(len(self.manager.sick_agents))
        for index, agent in enumerate(self.manager.sick_agents):
            if rolls[index] > agent.get_infection_ratio(self.manager.consts):
                v[agent.ID] = False

        u = self.manager.matrix.matrix.dot(v)
        infections = np.random.random(u.shape) < (1 - np.exp(u))
        caught_rolls = np.random.random(u.shape) < self.manager.consts.caught_sicks_ratio
        for agent, value, caught_roll in zip(self.manager.agents, infections, caught_rolls):
            if value and agent.infect(self.manager.step_counter):
                self.manager.sick_agents.add(agent)
                if caught_roll:
                    if self.manager.consts.home_quarantine_sicks:
                        self.agents_to_home_quarantine.append(agent)
                    elif self.manager.consts.full_quarantine_sicks:
                        self.agents_to_full_quarantine.append(agent)

    def _update_sick_agents(self):
        """
        after each day, go through all sick agents and updates their status (allowing them to recover or die)
        """
        # return parameters
        new_dead = 0
        new_recovered = 0

        to_remove = set()
        rolls = np.random.random(len(self.manager.sick_agents))
        # moved the code from agent.day_passed here, so that it will be more easily managed
        for agent, roll in zip(self.manager.sick_agents, rolls):
            if agent.infection_date is None or agent.infection_date < 0:
                continue
            if self.manager.step_counter == agent.infection_date \
                    + self.manager.consts.average_silent_time_days:
                if roll < self.manager.consts.Asymptomatic_ratio:
                    agent.change_medical_state(MedicalState.Asymptomatic)
                else:
                    agent.change_medical_state(MedicalState.Symptomatic)
                self.manager.sick_agent_vector[agent.ID] = True
            elif self.manager.step_counter == agent.infection_date \
                    + self.manager.consts.average_silent_time_days \
                    + self.manager.consts.average_sick_time_days:
                to_remove.add(agent)
                self.manager.sick_agent_vector[agent.ID] = False
                if roll < self.manager.consts.death_ratio:
                    agent.change_medical_state(MedicalState.Deceased)
                    new_dead = new_dead + 1
                else:
                    agent.change_medical_state(MedicalState.Immune)
                    new_recovered = new_recovered + 1
        self.manager.sick_agents.difference_update(to_remove)

        return new_dead, new_recovered
