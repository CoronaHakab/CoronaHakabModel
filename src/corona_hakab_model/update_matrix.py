from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np

from analyzers.state_machine_analysis import monte_carlo_state_machine_analysis
from common.social_circle import SocialCircle
from generation.connection_types import ConnectionTypes
from policies_manager import ConditionedPolicy

if TYPE_CHECKING:
    from manager import SimulationManager


class Policy:
    """
    This represents a policy. 
    """

    def __init__(self, connection_change_factor: float, conditions: Iterable[Callable[[Any], bool]]):
        self.factor = connection_change_factor
        self.conditions = conditions

    def check_applies(self, arg):
        applies = True
        for condition in self.conditions:
            applies = applies and condition(arg)
        return applies


class PolicyByCircles:
    def __init__(self, policy: Policy, circles: Iterable[SocialCircle]):
        self.circles = circles
        self.policy = policy


class UpdateMatrixManager:
    """
    Manages the "Update Matrix" stage of the simulation.
    """

    def __init__(self, manager: SimulationManager):  # noqa: F821 - todo how to fix it?
        self.manager = manager
        # unpacking commonly used information from manager
        self.matrix = manager.matrix
        self.matrix_type = manager.matrix_type
        self.depth = manager.depth
        self.logger = manager.logger
        self.consts = manager.consts
        self.size = len(manager.agents)
        # todo unpack more important information
        self.normalize_factor = None
        self.total_contagious_probability = None
        self.normalize()

    def normalize(self):
        """
        this function should normalize the weights within W to represent the infection rate.
        As r0=bd, where b is number of daily infections per person
        """
        self.logger.info(f"normalizing matrix")
        if self.normalize_factor is None:
            # updates r0 to fit the contagious length and ratio.
            population_size_for_mc = self.consts.population_size_for_state_machine_analysis
            machine_state_statistics = monte_carlo_state_machine_analysis(dict(population_size=population_size_for_mc))
            states_time = machine_state_statistics['state_duration_expected_time']
            total_contagious_probability = 0
            for state in self.manager.medical_machine.states:
                total_contagious_probability += states_time[state.name] * state.contagiousness.mean_val
            beta = self.consts.r0 / total_contagious_probability

            # saves this for the effective r0 graph
            self.total_contagious_probability = total_contagious_probability

            # Random connections
            random_total = np.dot(self.manager.num_of_random_connections.sum(0),
                                  self.manager.random_connections_strength)

            # this factor should be calculated once when the matrix is full, and be left un-changed for the rest of the run.
            self.normalize_factor = (beta * self.size) / (self.matrix.total() + random_total)

            # Normalize random connections (only once!)
            self.manager.random_connections_strength *= self.normalize_factor

        self.matrix *= self.normalize_factor  # now each entry in W is such that bd=R0

    def change_connections_policy(self, connection_types_to_use: Iterable[ConnectionTypes]):
        self.logger.info(f"changing policy. keeping all matrices of types: {connection_types_to_use}")
        factors = np.zeros(self.depth, dtype=np.float32)
        for connection_type in connection_types_to_use:
            ind = connection_type.value
            factors[ind] = 1
        self.matrix.set_factors(factors)
        self.normalize()

    def reset_agent(self, connection_type, index):
        self.matrix.reset_mul_row(connection_type, index)
        self.matrix.reset_mul_col(connection_type, index)
        self.manager.random_connections_factor[index, connection_type] = 1

    def factor_agent(self, index, connection_type, factor):
        self.matrix.mul_sub_row(connection_type, index, factor)
        self.matrix.mul_sub_col(connection_type, index, factor)
        self.manager.random_connections_factor[index, connection_type] *= factor

    def reset_policies_by_connection_type(self, connection_type):
        for i in range(self.size):
            if not self.manager.agents_in_isolation[i]:
                self.reset_agent(connection_type, i)

        # letting all conditioned policies acting upon this connection type know they are canceled
        for conditioned_policy in self.consts.connection_type_to_conditioned_policy[connection_type]:
            conditioned_policy.active = False

    def apply_policy_on_circles(self, policy: Policy, circles: Iterable[SocialCircle]):
        affected_circles = []

        # for now, we will not update the matrix at all
        for circle in circles:
            # check if circle is relevent to conditions
            flag = True
            for condition in policy.conditions:
                flag = flag and condition(circle)
            if not flag:
                # some condition returned False - skip circle
                continue

            affected_circles.append(circle)
            connection_type = circle.connection_type
            factor = policy.factor
            for agent in circle.agents:
                self.factor_agent(agent.index, connection_type, factor)

        return affected_circles

    def check_and_apply(
            self,
            con_type: ConnectionTypes,
            circles: Iterable[SocialCircle],
            conditioned_policy: ConditionedPolicy,
            **activating_condition_kwargs,
    ):
        affected_circles = []  # list of circles affected by activating the policy
        activating_policy = \
            (not conditioned_policy.active) and conditioned_policy.activating_condition(activating_condition_kwargs)
        if activating_policy:
            self.logger.info("activating policy on circles")
            self.reset_policies_by_connection_type(con_type)
            affected_circles = self.apply_policy_on_circles(conditioned_policy.policy, circles)
            conditioned_policy.active = True
            # adding the message
            self.manager.policy_manager.add_message_to_manager(conditioned_policy.message)
        return activating_policy, affected_circles

    def change_agent_relations_by_factor(self, agent, factor):
        for connection_type in ConnectionTypes:
            self.factor_agent(agent.index, connection_type, factor)

    def validate_matrix(self):
        submatrixes_rows_nonzero_columns = self.matrix.non_zero_columns()
        for rows_nonzero_columns in submatrixes_rows_nonzero_columns:
            for row_index, nonzero_columns in enumerate(rows_nonzero_columns):
                for column_index in nonzero_columns:
                    assert self.matrix.get(row_index, column_index) == self.matrix.get(column_index,
                                                                                       row_index), "Matrix is not symmetric"
                    assert 1 >= self.matrix.get(row_index,
                                                column_index) >= 0, "Some values in the matrix are not probabilities"
