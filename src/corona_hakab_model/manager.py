import logging
from collections import defaultdict
from typing import Callable, Iterable, List, Union

import infection
import update_matrix
from agent import InitialSickAgents
from agents_df import AgentsDf
from consts import Consts
from detection_model import healthcare
from detection_model.healthcare import PendingTestResult, PendingTestResults
from generation.circles_generator import PopulationData
from generation.matrix_generator import MatrixData
from medical_state_manager import MedicalStateManager
from policies_manager import PolicyManager
from state_machine import PendingTransfers
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
            consts: Consts = Consts(),
    ):
        # setting logger
        self.logger = logging.getLogger("simulation")
        logging.basicConfig()
        self.logger.setLevel(logging.INFO)
        self.logger.info("Creating a new simulation.")

        # unpacking data from generation
        agents = population_data.agents
        self.geographic_circles = population_data.geographic_circles
        self.social_circles_by_connection_type = population_data.social_circles_by_connection_type
        self.geographic_circle_by_agent_index = population_data.geographic_circle_by_agent_index
        self.social_circles_by_agent_index = population_data.social_circles_by_agent_index

        self.matrix_type = matrix_data.matrix_type
        self.matrix = matrix_data.matrix
        self.depth = matrix_data.depth

        # setting up medical things
        self.consts = consts
        self.medical_machine = consts.medical_state_machine()
        initial_state = self.medical_machine.initial
        self.agents_df = AgentsDf(agents, self.medical_machine)

        self.pending_transfers = PendingTransfers()

        # healthcare related data
        self.pending_test_results = PendingTestResults()

        # # initializing agents to current simulation
        # for agent in self.agents:
        #     agent.add_to_simulation(self, initial_state)
        initial_state.add_many(self.agents_df.agents_indexes())

        # initializing simulation modules
        self.simulation_progression = SimulationProgression([Supervisable.coerce(a, self) for a in supervisable_makers],
                                                            self)
        self.update_matrix_manager = update_matrix.UpdateMatrixManager(self)
        self.infection_manager = infection.InfectionManager(self)
        self.healthcare_manager = healthcare.HealthcareManager(self)
        self.medical_state_manager = MedicalStateManager(self)
        self.policy_manager = PolicyManager(self)

        self.current_step = 0

        # initializing data for supervising
        # dict(day:int -> message:string) saving policies messages
        self.policies_messages = defaultdict(str)
        self.initial_sick_agents = InitialSickAgents()

        self.new_sick_counter = 0
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

        # run infection
        new_sick = self.infection_manager.infection_step()

        # progress transfers
        self.medical_state_manager.step(new_sick)

        self.current_step += 1

        self.simulation_progression.snapshot(self)

    def progress_tests_and_isolation(self, new_tests: List[PendingTestResult]):
        self.new_detected_daily = 0
        new_results = self.pending_test_results.advance()
        for agent_ind, test_result, _ in new_results:
            if test_result:
                if not self.agents_df.ever_tested_positive(agent_ind):
                    # TODO: awful late night implementation, improve ASAP
                    self.new_detected_daily += 1
            self.agents_df.set_test_result(agent_ind, test_result)

        # TODO send the detected agents to the selected kind of isolation
        # TODO: Track isolated agents
        # TODO: Remove healthy agents from isolation?

        for new_test in new_tests:
            self.agents_df.set_test_start(new_test.agent_index, self.current_step)
            self.pending_test_results.append(new_test)

    def setup_sick(self):
        """"
        setting up the simulation with a given amount of infected people
        """
        # todo we only do this once so it's fine but we should really do something better
        agents_to_infect = list(range(self.consts.initial_infected_count))
        infected_state = self.medical_machine.default_state_upon_infection
        self.agents_df.change_agents_state(agents_to_infect, infected_state)

        self.initial_sick_agents.add_many_agents(self.agents_df.at(agents_to_infect))

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
        self.initial_sick_agents.export()
        for i in range(self.consts.total_steps):
            self.step()
            self.logger.info(f"performing step {i + 1}/{self.consts.total_steps}")

        # clearing lru cache after run
        # self.consts.medical_state_machine.cache_clear()
        Supervisable.coerce.cache_clear()

    def dump(self, **kwargs):
        return self.simulation_progression.dump(**kwargs)

    def __str__(self):
        return f"<SimulationManager: SIZE_OF_POPULATION={self.agents_df.n_agents()}, " f"STEPS_TO_RUN={self.consts.total_steps}>"
