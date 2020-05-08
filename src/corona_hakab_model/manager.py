import logging
from collections import defaultdict
from random import shuffle
from typing import Callable, Iterable, List, Union

import numpy as np

import infection
import update_matrix
from common.agent import SickAgents, InitialAgentsConstraints, Agent
from common.isolation_types import IsolationTypes
from common.detection_testing_types import PendingTestResult, PendingTestResults
from common.state_machine import PendingTransfers
from consts import Consts
from detection_model import healthcare
from generation.circles_generator import PopulationData
from generation.connection_types import ConnectionTypes
from generation.matrix_generator import MatrixData, ConnectionData
from medical_state_manager import MedicalStateManager
from policies_manager import PolicyManager
from supervisor import Supervisable, SimulationProgression


class SimulationManager:
    """
    A simulation manager is the main class, it manages the steps performed with policies
    """

    def __init__(

            self,
            supervisable_makers: Iterable[Union[str, Supervisable, Callable]],
            population_data: PopulationData,
            matrix_data: MatrixData,
            connection_data: ConnectionData,
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
        self.agents = np.array(population_data.agents)
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

        self.connection_data = connection_data

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
        self.agents_in_isolation = np.full(fill_value=IsolationTypes.NONE,
                                           shape=len(self.agents),
                                           dtype=IsolationTypes)
        self.agents_connections_coeffs = np.ones(shape=(len(self.agents), self.depth))
        self.date_of_last_test = np.zeros(len(self.agents), dtype=int)
        self.pending_test_results = PendingTestResults()
        self.step_to_isolate_agent = np.full(len(self.agents), -1, dtype=int)  # full of null step
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

        self.current_step = 0

        # initializing data for supervising
        # dict(day:int -> message:string) saving policies messages
        self.policies_messages = defaultdict(str)
        self.sick_agents = SickAgents()

        self.new_sick_counter = 0
        self.new_sick_by_infection_method = {connection_type: 0 for connection_type in ConnectionTypes}
        self.new_sick_by_infector_medical_state = {
                "Latent": 0,
                "Latent-Asymp": 0,
                "Latent-Presymp": 0,
                "AsymptomaticBegin": 0,
                "AsymptomaticEnd": 0,
                "Pre-Symptomatic": 0,
                "Mild-Condition-Begin": 0,
                "Mild-Condition-End": 0,
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
        self.progress_tests(new_tests)

        if self.consts.should_isolate_positive_detected:
            self.progress_isolations()

        self.new_sick_by_infection_method = {connection_type: 0 for connection_type in ConnectionTypes}
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

        self.current_step += 1

        self.simulation_progression.snapshot(self)

    def progress_isolations(self):
        detected_positive_now = self.tested_positive_vector & \
                                (self.date_of_last_test == self.current_step)
        detected_positive_to_isolate = detected_positive_now & (self.agents_in_isolation == IsolationTypes.NONE)
        detected_positive_indices = np.nonzero(detected_positive_to_isolate)[0]
        for agent_index in detected_positive_indices:
            # Get number of days agent is isolated
            if self.agents_in_isolation[agent_index] != IsolationTypes.NONE:
                # TODO: Need to take into account when got out of isolation also
                number_of_days_isolated = max(0,
                                              self.current_step - self.step_to_isolate_agent[agent_index])
            else:
                number_of_days_isolated = 0
            weekly_connection_isolation_ratio = 1 - min(number_of_days_isolated / 7, 1)
            for connected_agents in self.connection_data.connected_ids_by_strength[agent_index].values():
                for daily_agent_id in connected_agents.daily_connections:  # All of this need to be home isolated
                    # If were not isolated, and not sick, needs to
                    if self.agents_in_isolation[daily_agent_id] == IsolationTypes.NONE and \
                            not self.tested_positive_vector[daily_agent_id]:
                        if self.step_to_isolate_agent[daily_agent_id] < self.current_step:
                            self.step_to_isolate_agent[daily_agent_id] = self.current_step + \
                                                                         self.consts.isolate_after_num_day
                agents_to_iterate = list(filter(lambda index: (self.agents_in_isolation[index] == IsolationTypes.NONE)
                                                              and not self.tested_positive_vector[index],
                                                connected_agents.weekly_connections))
                non_sick_agents = self.medical_machine.default_state_upon_infection.agents
                shuffle(agents_to_iterate)
                agents_to_iterate.sort(key=lambda index: self.agents[index] in non_sick_agents, reverse=True)
                how_many_to_isolate = round(len(agents_to_iterate) * weekly_connection_isolation_ratio)
                j = 0
                for i in range(how_many_to_isolate):
                    # If already about to get isolated, do not update it
                    if self.step_to_isolate_agent[agents_to_iterate[i]] > self.current_step:
                        continue
                    self.step_to_isolate_agent[agents_to_iterate[i]] = self.current_step + \
                                                                       self.consts.isolate_after_num_day
                    j += 1
            # Isolate the agent
            self.step_to_isolate_agent[agent_index] = self.current_step + self.consts.isolate_after_num_day

        # TODO: Remove healthy agents from isolation?
        self.isolate_agents()

    def progress_tests(self, new_tests: List[PendingTestResult]):
        self.new_detected_daily = 0
        new_results = self.pending_test_results.advance()
        for agent, test_result, _ in new_results:
            self.new_detected_daily += test_result
            agent.set_test_result(test_result)

        for new_test in new_tests:
            new_test.agent.set_test_start()
            self.pending_test_results.append(new_test)

    def get_isolation_groups_by_reason(self, agent_to_group):
        tested_positive = list()
        first_circle = list()
        symptomatic = list()
        for agent in agent_to_group:
            if self.tested_positive_vector[agent.index]:
                tested_positive.append(agent)
            elif agent.medical_state.has_symptoms:
                symptomatic.append(agent)
            else:
                first_circle.append(agent)
        return dict(tested_positive=tested_positive,
                    first_circle=first_circle,
                    symptomatic=symptomatic)

    def isolate_agents(self):
        can_be_isolated = self.step_to_isolate_agent == self.current_step  # this is the day to isolate them
        remaining = self.agents[can_be_isolated]  # get those agents
        isolation_groups = self.get_isolation_groups_by_reason(remaining)
        remaining_sick_or_symp = isolation_groups['tested_positive'] + isolation_groups['symptomatic']
        remaining_healthy = isolation_groups['first_circle']
        # Sample who will obey
        # Cannot sample if array is of size 0
        if len(remaining_sick_or_symp) == 0:
            sick_or_symp_will_obey_isolation = np.empty((0,), dtype=Agent)
        else:
            sample_size = round(self.consts.sick_to_p_obey_isolation[True] *
                                len(remaining_sick_or_symp))
            sick_or_symp_will_obey_isolation = np.random.choice(remaining_sick_or_symp,
                                                                size=sample_size,
                                                                replace=False)
        if len(remaining_healthy) == 0:
            healthy_will_obey_isolation = np.empty((0,), dtype=Agent)
        else:
            sample_size = round(self.consts.sick_to_p_obey_isolation[False] *
                                len(remaining_healthy))
            healthy_will_obey_isolation = np.random.choice(remaining_healthy,
                                                           size=sample_size,
                                                           replace=False)

        with self.matrix.lock_rebuild():
            for agent in np.concatenate((sick_or_symp_will_obey_isolation,
                                         healthy_will_obey_isolation)):
                # keep track about who is in isolation and its type
                current_isolation_type = self.get_isolation_type(agent)
                isolation_factor = self.consts.isolation_factor[current_isolation_type]
                self.agents_in_isolation[agent.index] = current_isolation_type
                self.update_matrix_manager.change_agent_relations_by_factor(agent,
                                                                            isolation_factor)  # change the matrix


    def get_isolation_type(self, agent):  # TODO: not here...
        """
            Gets as input an agent and return the kind of isolation he should be in
        """
        if self.tested_positive_vector[agent.index]:
            return IsolationTypes.HOTEL
        return IsolationTypes.HOME

    def get_agents_out_of_isolation(self, agents_list: List):
        for agent in agents_list:
            for connection in ConnectionTypes:
                if self.agents_in_isolation[agent.index]:
                    current_row_factor = self.agents_connections_coeffs[agent.index, connection]
                    self.matrix.set_sub_row(connection, agent.index, current_row_factor)
                    self.matrix.set_sub_col(connection, agent.index, current_row_factor)
                    self.agents_in_isolation[agent.index] = IsolationTypes.NONE

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
