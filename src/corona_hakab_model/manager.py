import logging
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Union
from random import random

import healthcare
import infection
import numpy as np
import update_matrix
from consts import Consts
from generation.circles_generator import PopulationData
from generation.connection_types import ConnectionTypes
from generation.matrix_generator import MatrixData
from healthcare import PendingTestResult, PendingTestResults
from medical_state import MedicalState
from state_machine import PendingTransfers
from supervisor import Supervisable, Supervisor
from update_matrix import PolicyByCircles


class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    def __init__(
        self,
        supervisable_makers: Iterable[Union[str, Supervisable, Callable]],
        population_data: PopulationData,
        matrix_data: MatrixData,
        consts: Consts = Consts(),
    ):
        # setting logger
        self.logger = logging.getLogger("simulation")
        logging.basicConfig()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating a new simulation.")

        # unpacking data from generation
        self.agents = population_data.agents
        self.geographic_circles = population_data.geographic_circles
        self.social_circles_by_connection_type = population_data.social_circles_by_connection_type

        self.matrix_type = matrix_data.matrix_type
        self.matrix = matrix_data.matrix
        self.depth = matrix_data.depth

        # setting up medical things
        self.consts = consts
        self.medical_machine = consts.medical_state_machine()
        initial_state = self.medical_machine.initial

        self.pending_transfers = PendingTransfers()
        self.detected_daily = 0

        # the manager holds the vector, but the agents update it
        self.contagiousness_vector = np.zeros(len(self.agents), dtype=float)  # how likely to infect others
        self.susceptible_vector = np.zeros(len(self.agents), dtype=bool)  # can get infected

        # healthcare related data
        self.living_agents_vector = np.ones(len(self.agents), dtype=bool)
        self.test_willingness_vector = np.zeros(len(self.agents), dtype=float)
        self.tested_vector = np.zeros(len(self.agents), dtype=bool)
        self.tested_positive_vector = np.zeros(len(self.agents), dtype=bool)
        self.date_of_last_test = np.zeros(len(self.agents), dtype=int)
        self.pending_test_results = PendingTestResults()

        # initializing agents to current simulation
        for agent in self.agents:
            agent.add_to_simulation(self, initial_state)
        initial_state.add_many(self.agents)

        # initializing simulation modules
        self.supervisor = Supervisor([Supervisable.coerce(a, self) for a in supervisable_makers], self)
        self.update_matrix_manager = update_matrix.UpdateMatrixManager(self)
        self.infection_manager = infection.InfectionManager(self)
        self.healthcare_manager = healthcare.HealthcareManager(self)

        self.current_date = 0

        self.new_sick_counter = 0

        self.logger.info("Created new simulation.")

    def step(self):
        """
        run one step
        """
        # todo this does nothing right now.
        # update matrix
        self.update_matrix_manager.update_matrix_step()
        
        # change school openage policies
        self.change_school_openage()
        
        # run tests
        new_tests = self.healthcare_manager.testing_step(
            self.consts.detection_test, self.consts.daily_num_of_tests, self.consts.testing_policy
        )

        # progress tests and isolate the detected agents (update the matrix)
        self.progress_tests_and_isolation(new_tests)

        # run infection
        new_sick = self.infection_manager.infection_step()

        # progress transfers
        self.progress_transfers(new_sick)

        self.current_date += 1

        self.supervisor.snapshot(self)

    def progress_tests_and_isolation(self, new_tests: List[PendingTestResult]):
        agents_detected = []
        new_results = self.pending_test_results.advance()
        for agent, test_result, _ in new_results:
            agent.set_test_result(test_result)
            if test_result:
                agents_detected.append(agents_detected)

        self.detected_daily = len(agents_detected)

        # TODO send the detected agents to the selected kind of isolation
        # TODO: Track isolated agents
        # TODO: Remove healthy agents from isolation?

        for new_test in new_tests:
            new_test.agent.set_test_start()
            self.pending_test_results.append(new_test)

    def progress_transfers(self, new_sick: Dict[MedicalState, List]):
        # all the new sick agents are leaving their previous step
        changed_state_leaving = new_sick
        # agents which are going to enter the new state
        changed_state_introduced = defaultdict(list)
        # list of all the new sick agents
        new_sick_list = sum(changed_state_leaving.values(), [])

        # saves this number for supervising
        self.new_sick_counter = len(new_sick_list)
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
            
    def change_school_openage(self):
        if not Consts.should_change_school_openage or self.current_date not in Consts.school_openage_factors.keys():
            return 
        
        # first reset all schools
        self.update_matrix_manager.reset_policies_by_connection_type(ConnectionTypes.School)
        
        # create Policy object
        new_openage_factor = Consts.school_openage_factors[self.current_date]
        def should_open(): return random() > new_openage_factor
        policy = Policy(0, should_open) # 0 - school is closed
        self.update_matrix_manager.apply_policy_on_circles(policy, self.social_circles_by_connection_type[ConnectionTypes.School])

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
            if self.consts.active_isolation:
                if i == self.consts.stop_work_days:
                    self.update_matrix_manager.change_connections_policy(
                        {ConnectionTypes.Family, ConnectionTypes.Other}
                    )
                elif i == self.consts.resume_work_days:
                    self.update_matrix_manager.change_connections_policy(ConnectionTypes)
            self.step()
            self.logger.info(f"performing step {i + 1}/{self.consts.total_steps}")

        # clearing lru cache after run
        self.consts.medical_state_machine.cache_clear()
        Supervisable.coerce.cache_clear()

    def plot(self, **kwargs):
        self.supervisor.plot(**kwargs)

    def stackplot(self, **kwargs):
        self.supervisor.stack_plot(**kwargs)

    def __str__(self):
        return f"<SimulationManager: SIZE_OF_POPULATION={len(self.agents)}, " f"STEPS_TO_RUN={self.consts.total_steps}>"
