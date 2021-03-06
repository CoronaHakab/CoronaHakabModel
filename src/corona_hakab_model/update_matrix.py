from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import numpy as np

from analyzers.state_machine_analysis import monte_carlo_state_machine_analysis
from common.isolation_types import IsolationTypes
from common.social_circle import SocialCircle
from generation.connection_types import ConnectionTypes
from policies_manager import ConditionedPolicy, Policy

if TYPE_CHECKING:
    from manager import SimulationManager


class UpdateMatrixManager:
    """
    Manages the "Update Matrix" stage of the simulation.
    """

    def __init__(self, manager: SimulationManager):  # noqa: F821 - todo how to fix it?
        self.manager = manager
        # unpacking commonly used information from manager
        self.matrix = manager.matrix
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
            agents_ages = [_.age for _ in self.manager.agents]
            ages_array, ages_counts_array = np.unique(agents_ages, return_counts=True)
            age_dist = {age: count/len(agents_ages) for age, count in zip(ages_array, ages_counts_array)}
            state_machine_analysis_config = dict(age_distribution=age_dist,
                                                 population_size=population_size_for_mc)
            machine_state_statistics = monte_carlo_state_machine_analysis(state_machine_analysis_config)
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
        self.manager.agents_connections_coeffs[index, connection_type] = 1
        self.manager.random_connections_factor[index, connection_type] = 1

    def factor_agent(self, index, connection_type, factor):
        self.matrix.mul_sub_row(connection_type, index, factor)
        self.matrix.mul_sub_col(connection_type, index, factor)
        self.manager.agents_connections_coeffs[index, connection_type] *= factor
        self.manager.random_connections_factor[index, connection_type] *= factor

    def reset_policies_by_connection_type(self, connection_type, agents_ids_to_reset=None):
        if agents_ids_to_reset is None:
            agents_ids_to_reset = list(range(self.size))
        for i in agents_ids_to_reset:
            if self.manager.agents_in_isolation[i] != IsolationTypes.NONE:
                self.reset_agent(connection_type, i)
            else:  # When out of isolation, the policy is not applied on him.
                self.manager.agents_connections_coeffs[i, connection_type] = 1

        # letting all conditioned policies acting upon this connection type know they are canceled
        if connection_type in self.consts.connection_type_to_conditioned_policy:
            for conditioned_policy in self.consts.connection_type_to_conditioned_policy[connection_type]:
                conditioned_policy.active = False

    def apply_policy_on_circles(self, policy: Policy, circles: Iterable[SocialCircle]):
        affected_circles = []

        # for now, we will not update the matrix at all
        for circle in circles:
            if policy.check_applies_on_circle(circle):
                affected_circles.append(circle)
                connection_type = circle.connection_type
                factor = policy.factor
                for agent in circle.agents:
                    if policy.check_applies_on_agent(agent):
                        self.factor_agent(agent.index, connection_type, factor)
                        agent.policy_props.update(policy.policy_props_update)

        return affected_circles

    def apply_conditional_policy(
            self,
            con_type: ConnectionTypes,
            circles: Iterable[SocialCircle],
            conditioned_policy: ConditionedPolicy,
    ):
        self.logger.info(f"activating policy {conditioned_policy.message} on circles")
        if conditioned_policy.reset_current_limitations:
            self.reset_policies_by_connection_type(con_type)
        affected_circles = self.apply_policy_on_circles(conditioned_policy.policy, circles)
        conditioned_policy.active = True
        # adding the message
        self.manager.policy_manager.add_message_to_manager(conditioned_policy.message)
        return affected_circles

    def change_agent_relations_by_factor(self, agent, factor):
        try:
            float(factor)  # If the input is a number, we create dict with the factor
            factor = {connection: factor for connection in ConnectionTypes}
        except:  # If they do not succeed, proceed
            pass
        for connection_type, connection_factor in factor.items():
            self.factor_agent(agent.index, connection_type, connection_factor)

    def validate_matrix(self):
        submatrixes_rows_nonzero_columns = self.matrix.non_zero_columns()
        for rows_nonzero_columns in submatrixes_rows_nonzero_columns:
            for row_index, nonzero_columns in enumerate(rows_nonzero_columns):
                for column_index in nonzero_columns:
                    assert self.matrix.get(row_index, column_index) == self.matrix.get(column_index,
                                                                                       row_index), "Matrix is not symmetric"
                    assert 1 >= self.matrix.get(row_index,
                                                column_index) >= 0, "Some values in the matrix are not probabilities"