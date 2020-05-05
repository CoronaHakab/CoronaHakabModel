from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, List, Dict

from common.social_circle import SocialCircle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manager import SimulationManager

class PolicyManager:
    def __init__(self, manager: SimulationManager):
        self.manager = manager
        self.update_matrix_manager = manager.update_matrix_manager
        self.consts = manager.consts
        self.logger = manager.logger
        # For each applied policy, save a list of affected circles
        self.daily_affected_circles: Dict[ConditionedPolicy, List[SocialCircle]] = defaultdict(List)

    def perform_policies(self):
        """
        this methos will be called daily.
        checks if there is a new policy to act upon. if so, active it
        :return:
        """
        # Initialize list of daily affected circles
        self.daily_affected_circles = defaultdict(List)

        # checks for a matrices summing change policy
        if self.consts.change_policies and self.manager.current_step in self.consts.policies_changes:
            self.logger.info("changing policy")
            self.update_matrix_manager.change_connections_policy(
                self.consts.policies_changes[self.manager.current_step][0]
            )
            self.add_message_to_manager(self.consts.policies_changes[self.manager.current_step][1])

        # check for a partial opening policy
        if self.consts.partial_opening_active:
            self.activate_partial_opening_policies()

    def activate_partial_opening_policies(self):
        """
        the following policies are performing on a specific connection type.
        they apply a given factor for each agent in some circles, of the same connection type
        the circles to act upon are chosen using a lambda, defined in consts
        :return:
        """
        for con_type, conditioned_policies in self.consts.connection_type_to_conditioned_policy.items():
            circles = self.manager.social_circles_by_connection_type[con_type]
            # going through each policy activator.
            conditional_policies_to_activate = []
            for cond_policy in conditioned_policies:
                if self.update_matrix_manager.should_apply_policy(cond_policy, self.manager):
                    conditional_policies_to_activate.append(cond_policy)

            if len(conditional_policies_to_activate) > 0:
                # Prevent rebuild of each row after each action.
                # This line speeds up the policy activation part EXTREMELY!
                with self.update_matrix_manager.matrix.lock_rebuild():
                    for conditioned_policy in conditional_policies_to_activate:
                        affected_circles = self.update_matrix_manager.apply_conditional_policy(
                            con_type, circles, conditioned_policy)
                        # Append affected circles to the daily affected circles
                        self.update_daily_affected_circles(affected_circles, conditioned_policy)
                # When exiting the "with" context, the matrix will be completely rebuilt,
                # after all the new coefficients were set

    def update_daily_affected_circles(self, affected_circles, conditioned_policy):
        if conditioned_policy in self.daily_affected_circles:
            self.logger.info(f"Policy {conditioned_policy} was executed twice on the same day!")
            return
        self.daily_affected_circles[conditioned_policy] = affected_circles

    def add_message_to_manager(self, message: str):
        if message == "":
            return
        current_step = self.manager.current_step
        if current_step in self.manager.policies_messages:
            self.manager.policies_messages[current_step] += " " + message
        else:
            self.manager.policies_messages[current_step] = message


class Policy:
    """
    This represents a policy.
    """

    def __init__(
            self,
            connection_change_factor: float,
            circle_conditions: Iterable[Callable[[Any], bool]],
            agent_conditions: Iterable[Callable[[Any], bool]] = None,
            policy_props_update: Dict = None,

    ):

        self.factor = connection_change_factor
        self.circle_conditions = circle_conditions

        if policy_props_update is None:
            policy_props_update = dict()

        if agent_conditions is None:
            agent_conditions = []

        self.agent_conditions = agent_conditions
        self.policy_props_update = policy_props_update

    @staticmethod
    def _check_applies(conditions, arg):
        applies = True
        for condition in conditions:
            applies = applies and condition(arg)
        return applies

    def check_applies_on_circle(self, circle):
        return self._check_applies(self.circle_conditions, circle)

    def check_applies_on_agent(self, agent):
        return self._check_applies(self.agent_conditions, agent)


class PolicyByCircles:
    def __init__(self, policy: Policy, circles: Iterable[SocialCircle]):
        self.circles = circles
        self.policy = policy


class ConditionedPolicy:
    """
    this class contains a policy that is supposed to run when a given condition is satisfied.
    """

    __slots__ = "activating_condition", "policy", "active", "dont_repeat_while_active", "reset_current_limitations", "message"

    def __init__(
            self, activating_condition: Callable[[SimulationManager], bool], policy: Policy,
            reset_current_limitations=True,
            dont_repeat_while_active=True,
            active=False,
            message=""
    ):
        self.activating_condition = activating_condition
        self.policy = policy
        self.dont_repeat_while_active = dont_repeat_while_active
        self.reset_current_limitations = reset_current_limitations
        self.message = message
        self.active = active  # Is the policy currently active
