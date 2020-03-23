import numpy as np
import plot_stats
import corona_stats
from medical_state import MedicalState



class InfectionManager:
    """
    Manages the infection stage
    """
    
    def __init__(self, manager):
        self.agents_to_home_quarantine = []
        self.agents_to_full_quarantine = []
        self.manager = manager # more comfetable passing the manager object once. this passes a refrence so it will always be updated

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

        #a patch just for the mvp, in order to simulate infection ratio
        rolls = np.random.random(len(self.manager.sick_agents))
        for index,agent in enumerate(self.manager.sick_agents):
            if rolls[index] > agent.get_infection_ratio():
                v[agent.ID] = False

        u = self.manager.matrix.matrix.dot(v)
        infections = np.random.random(u.shape) < (1 - np.exp(u))
        for agent, value in zip(self.manager.agents, infections):
            if value and agent.infect(self.manager.step_counter):
                self.manager.sick_agents.add(agent)
                if plot_stats.home_quarantine_sicks and np.random.random() < plot_stats.precents_of_caught_sicks:  # todo switch to a specific distribution
                    self.agents_to_home_quarantine.append(agent)
                if plot_stats.full_quarantine_sicks and np.random.random() < plot_stats.precents_of_caught_sicks:
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
        for agent, roll in zip(self.manager.sick_agents, rolls): # moved the code from agent.day_passed here, so that it will be more easily managed
            if agent.infection_date is None or agent.infection_date < 0:
                continue
            if self.manager.step_counter == agent.infection_date + corona_stats.average_silent_time:  # todo use random with a given deviation
                if roll < corona_stats.Asymptomatic_ratio:
                    agent.change_medical_state(MedicalState.Asymptomatic)
                else:
                    agent.change_medical_state(MedicalState.Symptomatic)
                self.manager.sick_agent_vector[agent.ID] = True
            elif self.manager.step_counter == agent.infection_date + corona_stats.average_silent_time + corona_stats.average_sick_time: # todo use random with a given deviation
                to_remove.add(agent)
                self.manager.sick_agent_vector[agent.ID] = False
                if roll < corona_stats.death_ratio:
                    agent.change_medical_state(MedicalState.Deceased)
                    new_dead = new_dead + 1
                else:
                    agent.change_medical_state(MedicalState.Immune)
                    new_recovered = new_recovered + 1
        self.manager.sick_agents.difference_update(to_remove)

        return new_dead, new_recovered
