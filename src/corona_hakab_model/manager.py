import logging
from collections import defaultdict, Counter
from itertools import groupby, chain
from numpy.random import shuffle
from typing import Callable, Iterable, List, Union

import numpy as np

import infection
import update_matrix
from common.agent import SickAgents, InitialAgentsConstraints, Agent
from common.isolation_types import IsolationTypes
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

        self.matrix = matrix_data._matrix
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
        self.consecutive_negative_tests = Counter()
        self.ever_tested_positive_vector = np.zeros(len(self.agents), dtype=bool)
        self.agents_in_isolation = np.full(fill_value=IsolationTypes.NONE,
                                           shape=len(self.agents),
                                           dtype=IsolationTypes)
        self.agents_connections_coeffs = np.ones(shape=(len(self.agents), self.depth))
        self.date_of_last_test = np.zeros(len(self.agents), dtype=int)
        self.step_to_isolate_agent = np.full(len(self.agents), -1, dtype=np.int32)  # full of null step
        self.step_to_free_agent = np.full(len(self.agents), -1, dtype=np.int32)  # full of null step
        self.left_isolation_by_reason = Counter()

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

        self.logger.info("Created new simulation.")
        self.simulation_progression.snapshot(self)

    def step(self):
        """
        run one step
        """

        # checks if there is a policy to active.
        self.policy_manager.perform_policies()

        self.healthcare_manager.step()

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
        # Need to isolate every non-hotel isolated verified infected agent
        detected_positive_indices = [agent for agent in self.healthcare_manager.positive_detected_today
                                     if self.agents_in_isolation[agent] != IsolationTypes.HOTEL]
        # This is pretty fast even for large number of agents.
        # It removes the need to activate step_to_isolate_dist over and over again
        days_to_enter_isolation = self.consts.step_to_isolate_dist(size=len(self.agents))
        for agent_index in detected_positive_indices:
            # Get number of days agent is isolated
            if self.agents_in_isolation[agent_index] != IsolationTypes.NONE:
                # TODO: Need to take into account when got out of isolation also
                number_of_days_isolated = self.current_step - self.step_to_isolate_agent[agent_index]
            else:
                number_of_days_isolated = 0
            weekly_connection_isolation_ratio = 1 - min(number_of_days_isolated / 7, 1)

            # Isolate the agent
            self.step_to_isolate_agent[agent_index] = self.current_step + days_to_enter_isolation[agent_index]
            if not self.consts.isolate_first_circle:
                continue
            days_to_be_isolated = self.consts.home_isolation_time_bound - number_of_days_isolated
            for connection, connected_agents in self.connection_data.connected_ids_by_strength[agent_index].items():
                # need to be home isolated
                days_to_isolate_for_conn = self.consts.home_isolation_time_bound \
                    if connection == ConnectionTypes.Family \
                    else days_to_be_isolated
                for daily_agent_index in connected_agents.daily_connections:
                    # Need to isolate, ones that are not isolated/about.
                    # If tested positive, does not need further isolation
                    if self.step_to_isolate_agent[daily_agent_index] < self.current_step and \
                            self.agents_in_isolation[daily_agent_index] == IsolationTypes.NONE and \
                            daily_agent_index not in detected_positive_indices:
                        self.step_to_isolate_agent[daily_agent_index] = self.current_step + \
                                                                        days_to_enter_isolation[daily_agent_index]
                        self.step_to_free_agent[daily_agent_index] = max(self.step_to_isolate_agent[daily_agent_index] +
                                                                         days_to_isolate_for_conn,
                                                                         self.step_to_free_agent[daily_agent_index])

                agents_to_iterate, how_many_to_isolate = self._get_sorted_weekly_to_isolate(
                    connected_agents.weekly_connections,
                    weekly_connection_isolation_ratio)
                # For every agent we isolate, we randomly choose when he met the agent
                # Then, we need
                days_to_isolate = self.consts.home_isolation_time_bound - np.random.randint(low=number_of_days_isolated,
                                                                                            high=7,
                                                                                            size=how_many_to_isolate)
                for index, weekly_agent_index in enumerate(agents_to_iterate):
                    if index == how_many_to_isolate:
                        break
                    # If already about to get isolated or is isolated, do not update it
                    # Do not isolate again those that got tested right now
                    if self.step_to_isolate_agent[weekly_agent_index] < self.current_step and \
                            weekly_agent_index not in detected_positive_indices:
                        continue
                    self.step_to_isolate_agent[weekly_agent_index] = self.current_step + \
                                                                     days_to_enter_isolation[weekly_agent_index]
                    self.step_to_free_agent[weekly_agent_index] = self.step_to_isolate_agent[weekly_agent_index] + \
                                                                  days_to_isolate[index]

        # Isolating symptomatic agents
        if self.consts.isolate_symptomatic:
            for agent in self.medical_state_manager.new_agents_with_symptoms:
                # If is not getting ready to be isolated, or isolated already, then isolate
                if self.step_to_isolate_agent[agent.index] < self.current_step and \
                        self.agents_in_isolation[agent.index] == IsolationTypes.NONE and \
                        agent.index not in detected_positive_indices:
                    self.step_to_isolate_agent[agent.index] = self.current_step + days_to_enter_isolation[agent.index]

        # TODO: Remove healthy agents from isolation?
        self.isolate_agents()
        self.free_isolated_agents()

    def free_isolated_agents(self):
        agents_to_free = np.flatnonzero(self.step_to_free_agent == self.current_step)

        for agent in agents_to_free:
            for connection in ConnectionTypes:
                if self.agents_in_isolation[agent]:
                    current_row_factor = self.agents_connections_coeffs[agent, connection]
                    self.matrix.set_sub_row(connection, int(agent), current_row_factor)
                    self.matrix.set_sub_col(connection, int(agent), current_row_factor)

        self.step_to_isolate_agent[agents_to_free] = -1
        self.agents_in_isolation[agents_to_free] = IsolationTypes.NONE
        self.step_to_free_agent[agents_to_free] = -1
        self.left_isolation_by_reason['negative_tests'] = len(self.healthcare_manager.freed_neg_tested)
        self.left_isolation_by_reason['due_date'] = len(set(agents_to_free).difference(
            self.healthcare_manager.freed_neg_tested))

    def _get_sorted_weekly_to_isolate(self, weekly_connection, isolation_ratio):
        agents_to_iterate = [agent_id for agent_id in weekly_connection
                             if self.agents_in_isolation[agent_id] == IsolationTypes.NONE]
        non_infected_agents = self.medical_machine.default_state_upon_infection.ever_visited
        # We group to infected and non-infected
        how_many_to_isolate = int(round(len(agents_to_iterate) * isolation_ratio))
        agents_to_iterate_grouped_temp = list(groupby(agents_to_iterate,
                                                      key=lambda y: self.agents[y] in non_infected_agents))
        infected_agents_grouped = chain(
            *[group_iter[1] for group_iter in agents_to_iterate_grouped_temp if group_iter[1]])
        healthy_agents_grouped = chain(
            *[group_iter[1] for group_iter in agents_to_iterate_grouped_temp if not group_iter[1]])
        agents_to_iterate = chain(infected_agents_grouped, healthy_agents_grouped)
        return agents_to_iterate, how_many_to_isolate

    def get_isolation_groups_by_reason(self, agents_to_group):
        tested_positive = list()
        first_circle = list()
        symptomatic = list()
        for agent in agents_to_group:
            if self.ever_tested_positive_vector[agent.index]:
                # If until isolation starts, got negative answer
                # do not change isolation status
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

        for agent in np.concatenate((sick_or_symp_will_obey_isolation,
                                     healthy_will_obey_isolation)):
            # keep track about who is in isolation and its type
            current_isolation_type = self.get_isolation_type(agent)
            isolation_factor = self.consts.isolation_factor[current_isolation_type]
            self.update_matrix_manager.change_agent_relations_by_factor(agent,
                                                                        isolation_factor)  # change the matrix
            self.agents_in_isolation[agent.index] = current_isolation_type

    def get_isolation_type(self, agent):  # TODO: not here...
        """
            Gets as input an agent and return the kind of isolation he should be in
        """
        # This is OK, since once getting sick, you won't need to be isolated once you recover
        if self.ever_tested_positive_vector[agent.index]:
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
