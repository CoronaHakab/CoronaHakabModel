import logging
from collections import defaultdict
from random import shuffle
from typing import Callable, Iterable, List, Union
import numpy as np

import infection
import update_matrix
from common.agent import SickAgents, InitialAgentsConstraints
from consts import Consts
from detection_model import healthcare
from common.detection_testing_types import PendingTestResult, PendingTestResults
from generation.circles_generator import PopulationData
from generation.matrix_generator import MatrixData
from medical_state_manager import MedicalStateManager
from policies_manager import PolicyManager
from common.state_machine import PendingTransfers
from supervisor import Supervisable, SimulationProgression
from generation.connection_types import ConnectionTypes


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
            consts: Consts = Consts(),
    ):
        # setting logger
        self.logger = logging.getLogger("simulation")
        logging.basicConfig()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating a new simulation.")

        # unpacking data from generation
        self.initial_agent_constraints = inital_agent_constraints
        self.agents = population_data.agents
        self.geographic_circles = population_data.geographic_circles
        self.social_circles_by_connection_type = population_data.social_circles_by_connection_type
        self.geographic_circle_by_agent_index = population_data.geographic_circle_by_agent_index
        self.social_circles_by_agent_index = population_data.social_circles_by_agent_index
        self.num_of_random_connections = population_data.num_of_random_connections
        self.random_connections_strength = population_data.random_connections_strength
        self.random_connections_factor = np.ones_like(self.num_of_random_connections, dtype=float)

        self.matrix_type = matrix_data.matrix_type
        self.matrix = matrix_data.matrix
        self.depth = matrix_data.depth

        self.run_args = run_args

        # setting up medical things
        self.consts = consts
        self.medical_machine = consts.medical_state_machine()
        initial_state = self.medical_machine.initial

        self.pending_transfers = PendingTransfers()

        # the manager holds the vector, but the agents update it
        self.contagiousness_vector = np.zeros(len(self.agents), dtype=float)  # how likely to infect others
        self.susceptible_vector = np.zeros(len(self.agents), dtype=bool)  # can get infected

        # healthcare related data
        self.living_agents_vector = np.ones(len(self.agents), dtype=bool)
        self.test_willingness_vector = np.zeros(len(self.agents), dtype=float)
        self.tested_vector = np.zeros(len(self.agents), dtype=bool)
        self.tested_positive_vector = np.zeros(len(self.agents), dtype=bool)
        self.ever_tested_positive_vector = np.zeros(len(self.agents), dtype=bool)
        self.agents_in_isolation = np.zeros(len(self.agents), dtype=bool)
        self.agents_connections_factors = np.ones(shape=(len(self.agents), self.depth))
        self.date_of_last_test = np.zeros(len(self.agents), dtype=int)
        self.pending_test_results = PendingTestResults()
        self.step_to_isolate_agent = np.full(len(self.agents), -1, dtype=int)  # full of null step

        self.current_step = 0

        # initializing agents to current simulation
        for agent in self.agents:
            agent.add_to_simulation(self, initial_state)
        initial_state.add_many(self.agents)

        # initializing simulation modules
        self.simulation_progression = SimulationProgression([Supervisable.coerce(a, self) for a in supervisable_makers],
                                                            self)
        self.update_matrix_manager = update_matrix.UpdateMatrixManager(self)
        if run_args.validate_matrix:
            self.update_matrix_manager.validate_matrix()
        self.infection_manager = infection.InfectionManager(self)
        self.healthcare_manager = healthcare.HealthcareManager(self)
        self.medical_state_manager = MedicalStateManager(self)
        self.policy_manager = PolicyManager(self)

        # initializing data for supervising
        # dict(day:int -> message:string) saving policies messages
        self.policies_messages = defaultdict(str)
        self.sick_agents = SickAgents()

        self.new_sick_counter = 0
        self.new_sick_by_infection_method = {connection_type : 0 for connection_type in ConnectionTypes}
        self.new_sick_by_infector_medical_state = {
                "Latent": 0,
                "Latent-Asymp": 0,
                "Latent-Presymp": 0,
                "AsymptomaticBegin": 0,
                "AsymptomaticEnd": 0,
                "Pre-Symptomatic": 0,
                "Mild-Condition": 0,
                "NeedOfCloseMedicalCare": 0,
                "NeedICU": 0,
                "ImprovingHealth": 0,
                "PreRecovered": 0
        }
        self.new_detected_daily = 0

        self.logger.info("Created new simulation.")
        self.simulation_progression.snapshot(self)

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

        self.new_sick_by_infection_method = {connection_type : 0 for connection_type in ConnectionTypes}
        self.new_sick_by_infector_medical_state = defaultdict(int)
        # run infection
        new_infection_cases = self.infection_manager.infection_step()
        for agent, new_infection_case in new_infection_cases.items():
            self.sick_agents.add_agent(agent.get_snapshot())

            if self.consts.backtrack_infection_sources:
                self.new_sick_by_infection_method[new_infection_case.connection_type] += 1
                self.new_sick_by_infector_medical_state[new_infection_case.infector_agent.medical_state.name] += 1
        
        # progress transfers
        medical_machine_step_result = self.medical_state_manager.step(new_infection_cases.keys())
        self.new_sick_counter = medical_machine_step_result['new_sick']

        for agent in self.agents:
            agent.update_contagiousness()

        self.current_step += 1

        self.simulation_progression.snapshot(self)

    def progress_tests_and_isolation(self, new_tests: List[PendingTestResult]):
        self.new_detected_daily = 0
        new_results = self.pending_test_results.advance()
        for agent, test_result, _ in new_results:
            if test_result:
                if not self.ever_tested_positive_vector[agent.index]:
                    # TODO: awful late night implementation, improve ASAP
                    # set isolation date
                    self.step_to_isolate_agent[agent.index] = self.current_step + self.consts.isolate_after_num_day
                    self.new_detected_daily += 1

            agent.set_test_result(test_result)

        # TODO: Remove healthy agents from isolation?
        if self.consts.should_isolate_positive_detected:
            self.isolate_agents()

        for new_test in new_tests:
            new_test.agent.set_test_start()
            self.pending_test_results.append(new_test)

    def isolate_agents(self):
        can_be_isolated = self.step_to_isolate_agent == self.current_step  # this is the day to isolate them
        remaining = np.array(self.agents)[can_be_isolated]  # get those agents
        num_of_obedients = round(self.consts.p_will_obey_isolation * len(remaining))  # get number of agents to sample
        will_obey_isolation = np.random.choice(remaining, num_of_obedients, replace=False)  # sample those who will obey

        for agent in will_obey_isolation:
            self.agents_in_isolation[agent.index] = True  # keep track about who is in isolation
            self.update_matrix_manager.change_agent_relations_by_factor(agent,
                                                                        self.consts.isolation_factor)  # change the matrix

    def setup_sick(self):
        """"
        setting up the simulation with a given amount of infected people
        """
        agents_to_infect = []
        agent_index = 0
        agent_permutation = list(range(len(self.agents)))

        if self.run_args.randomize:
            self.logger.info("creating permutation")
            shuffle(agent_permutation)  # this is somewhat expensive for large sets, but imho it's worth it.
            self.logger.info("finished permuting")
        else:
            self.logger.info("running without permutation")
        if self.initial_agent_constraints.constraints is not None \
                and len(self.initial_agent_constraints.constraints) != self.consts.initial_infected_count:
            raise ValueError("Constraints file row number must match number of sick agents in simulation")
        while len(agents_to_infect) < self.consts.initial_infected_count:
            if agent_index == len(self.agents):
                raise ValueError("Initial sick agents over-constrained, couldn't find compatible agents")
            temp_agent = self.agents[agent_permutation[agent_index]]
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
