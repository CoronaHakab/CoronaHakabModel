import logging
from typing import Any, Callable, Iterable, List

import numpy as np
from generation.circles import SocialCircle
from generation.connection_types import ConnectionTypes


class PolicyManager:
    def __init__(self, manager: "SimulationManager"):
        self.manager = manager
        self.update_matrix_manager = manager.update_matrix_manager
        self.consts = manager.consts
        self.logger = manager.logger

    def perform_policies(self):
        """
        this methos will be called daily.
        checks if there is a new policy to act upon. if so, active it
        :return:
        """
        # checks for a matrices summing change policy
        if self.consts.change_policies and self.manager.current_step in self.consts.policies_changes:
            self.logger.info("changing policy")
            self.update_matrix_manager.change_connections_policy(
                self.consts.policies_changes[self.manager.current_step][0]
            )
            self.add_message_to_manager(self.consts.policies_changes[self.manager.current_step][1])

        # check for a partial opening policy
        if self.consts.partial_opening_active:
            self.active_partial_opening_policies()

    def active_partial_opening_policies(self):
        """
        this policies refers to policies acting on a specific connection type.
        those policies apllies a given factor for each agent in some circles, of the same connection type
        the circles to act upon are choosen using a lambda, defined in consts
        :return:
        """
        for con_type, conditioned_policies in self.consts.connection_type_to_conditioned_policy.items():
            circles = self.manager.social_circles_by_connection_type[con_type]
            # going through each policy activator.
            for conditioned_policy in conditioned_policies:
                # check if temp is satisfied
                self.update_matrix_manager.check_and_apply(con_type, circles, conditioned_policy, manager=self.manager)

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


class ConditionedPolicy:
    """
    this class contains a policy that is supposed to run when a given condition is satisfied.
    """

    __slots__ = "activating_condition", "policy", "active", "message"

    def __init__(self, activating_condition: Callable[[Any], bool], policy: Policy, active=False, message=""):
        self.activating_condition = activating_condition
        self.policy = policy
        self.active = active
        self.message = message
