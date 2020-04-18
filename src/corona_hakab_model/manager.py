import logging
from collections import defaultdict

from random import random, choice
from typing import Callable, Dict, Iterable, List, Union
import infection
import update_matrix
import numpy as np
from typing import Callable, Iterable, List, Union
import infection
import update_matrix
from agent import SickAgents, InitialAgentsConstraints
from consts import Consts
from detection_model import healthcare
from detection_model.healthcare import PendingTestResult, PendingTestResults
from generation.circles_generator import PopulationData
from generation.matrix_generator import MatrixData
from medical_state_manager import MedicalStateManager
from policies_manager import PolicyManager
from state_machine import PendingTransfers
from supervisor import Supervisable, SimulationProgression
from update_matrix import Policy



class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    def __init__(

            self,
            supervisable_makers: Iterable[Union[str, Supervisable, Callable]],
            population_data: PopulationData,
            matrix_data: MatrixData,
            inital_agent_constraints: InitialAgentsConstraints,
            run_args,
            consts: Consts = Consts()):

        # setting logger
        self.logger = logging.getLogger("simulation")
        logging.basicConfig()
        if not run_args.silent:
            self.logger.setLevel(logging.INFO)
        self.logger.info("Creating a new simulation.")

        # unpacking data from generation
        self.initial_agent_constraints = inital_agent_constraints
        self.agents = population_data.agents
        self.geographic_circles = population_data.geographic_circles
        self.social_circles_by_connection_type = population_data.social_circles_by_connection_type
        self.geographic_circle_by_agent_index = population_data.geographic_circle_by_agent_index
        self.social_circles_by_agent_index = population_data.social_circles_by_agent_index

        self.matrix_type = matrix_data.matrix_type
        self.matrix = matrix_data.matrix
        self.depth = matrix_data.depth

        self.run_args = run_args

        # setting up medical things
        self.consts = consts

        self.pending_transfers = PendingTransfers()

        self.update_matrix_manager = None
        self.reset()

        self.simulation_progression = SimulationProgression([Supervisable.coerce(a, self) for a in supervisable_makers],
                                                            self)
        self.logger.info("Created new simulation.")
        self.simulation_progression.snapshot(self)

    def reset(self):
        self.consts.medical_state_machine.cache_clear()  # In order to get the new medical_state_machine
        self.medical_machine = self.consts.medical_state_machine()
        initial_state = self.medical_machine.initial
        # the manager holds the vector, but the agents update it
        self.contagiousness_vector = np.zeros(len(self.agents), dtype=float)  # how likely to infect others
        self.susceptible_vector = np.zeros(len(self.agents), dtype=bool)  # can get infected

        # healthcare related data
        self.living_agents_vector = np.ones(len(self.agents), dtype=bool)
        self.test_willingness_vector = np.zeros(len(self.agents), dtype=float)
        self.tested_vector = np.zeros(len(self.agents), dtype=bool)
        self.tested_positive_vector = np.zeros(len(self.agents), dtype=bool)
        self.ever_tested_positive_vector = np.zeros(len(self.agents), dtype=bool)
        self.date_of_last_test = np.zeros(len(self.agents), dtype=int)
        self.pending_test_results = PendingTestResults()

        # initializing agents to current simulation
        for agent in self.agents:
            agent.add_to_simulation(self, initial_state)
        initial_state.add_many(self.agents)

        # initializing simulation modules
        if self.update_matrix_manager:
            factor = self.update_matrix_manager.normalize_factor
            self.update_matrix_manager = update_matrix.UpdateMatrixManager(manager=self,
                                                                           normalize_factor=factor)
        self.infection_manager = infection.InfectionManager(self)
        self.healthcare_manager = healthcare.HealthcareManager(self)
        self.medical_state_manager = MedicalStateManager(sim_manager=self)
        self.policy_manager = PolicyManager(self)

        self.current_step = 0

        # initializing data for supervising
        # dict(day:int -> message:string) saving policies messages
        self.policies_messages = defaultdict(str)
        self.sick_agents = SickAgents()

        self.new_sick_counter = 0
        self.new_detected_daily = 0

        self.logger.info("Done reset to simulation.")

    def step(self):
        """
        run one step
        """
        # checks if there is a policy to active.
        self.policy_manager.perform_policies()

        # run tests
        new_tests = self.healthcare_manager.testing_step()

        # progress tests and isolate the detected agents (update the matrix)
        self.progress_tests_and_isolation(new_tests)

        # run infection
        new_sick = self.infection_manager.infection_step()
        for agent in new_sick:
            self.sick_agents.add_agent(agent.get_snapshot())

        # progress transfers
        medical_machine_step_result = self.medical_state_manager.step(new_sick)
        self.new_sick_counter = medical_machine_step_result['new_sick']

        self.current_step += 1

        self.simulation_progression.snapshot(self)

    def progress_tests_and_isolation(self, new_tests: List[PendingTestResult]):
        self.new_detected_daily = 0
        new_results = self.pending_test_results.advance()
        for agent, test_result, _ in new_results:
            if test_result:
                if not self.ever_tested_positive_vector[agent.index]:
                    # TODO: awful late night implementation, improve ASAP
                    self.new_detected_daily += 1
                # if tested positive then isolate agent
                if self.consts.should_isolate_positive_detected:
                    self.update_matrix_manager.apply_full_isolation_on_agent(agent)

            agent.set_test_result(test_result)

        # TODO send the detected agents to the selected kind of isolation
        # TODO: Track isolated agents
        # TODO: Remove healthy agents from isolation?

        for new_test in new_tests:
            new_test.agent.set_test_start()
            self.pending_test_results.append(new_test)

    def setup_sick(self):
        """"
        setting up the simulation with a given amount of infected people
        """
        agents_to_infect = []
        agent_index = 0
        if self.initial_agent_constraints.constraints is not None\
                and len(self.initial_agent_constraints.constraints) != self.consts.initial_infected_count:
            raise ValueError("Constraints file row number must match number of sick agents in simulation")
        while len(agents_to_infect) < self.consts.initial_infected_count:
            if agent_index == len(self.agents):
                raise ValueError("Initial sick agents over-constrained, couldn't find compatible agents")
            temp_agent = self.agents[agent_index]
            agent_index += 1
            if self.initial_agent_constraints.constraints is None:
                agents_to_infect.append(temp_agent)
            else:
                for constraint in self.initial_agent_constraints.constraints:
                    if constraint.meets_constraint(temp_agent.get_snapshot()):
                        self.initial_agent_constraints.constraints.remove(constraint)
                        agents_to_infect.append(temp_agent)
                        break

        for agent in agents_to_infect:
            agent.set_medical_state_no_inform(self.medical_machine.default_state_upon_infection)
            self.sick_agents.add_agent(agent.get_snapshot())

        self.medical_machine.initial.remove_many(agents_to_infect)
        self.medical_machine.default_state_upon_infection.add_many(agents_to_infect)

        # take list of agents and create a pending transfer from their initial state to the next state
        self.medical_state_manager.pending_transfers.extend(
            self.medical_machine.default_state_upon_infection.transfer(agents_to_infect)
        )

    def run(self):
        """
        runs full simulation
        """
        self.setup_sick()
        if self.run_args.initial_sick_agents_path:
            self.sick_agents.export(self.run_args.initial_sick_agents_path)
        for i in range(self.consts.total_steps):
            self.step()
            self.logger.info(f"performing step {i + 1}/{self.consts.total_steps}")
        if self.run_args.all_sick_agents_path:
            self.sick_agents.export(self.run_args.all_sick_agents_path)

        # clearing lru cache after run
        # self.consts.medical_state_machine.cache_clear()
        Supervisable.coerce.cache_clear()

    def dump(self, **kwargs):
        return self.simulation_progression.dump(**kwargs)

    def __str__(self):
        return f"<SimulationManager: SIZE_OF_POPULATION={len(self.agents)}, " f"STEPS_TO_RUN={self.consts.total_steps}>"
