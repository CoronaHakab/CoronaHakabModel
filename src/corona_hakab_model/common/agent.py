from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
import pandas as pd
from numpy import nan

from .social_circle import SocialCircleConstraint
from generation.connection_types import ConnectionTypes
from .util import parse_str_to_num

if TYPE_CHECKING:
    from .medical_state import MedicalState
    from manager import SimulationManager


class Agent:
    """
    This class represents a person in our doomed world.
    """

    __slots__ = ("index", "medical_state", "manager", "age", "policy_props")

    # todo note that this changed to fit generation. should update simulation manager accordingly
    def __init__(self, index, age=None):
        self.index = index
        self.age = age
        # don't know if this is necessary
        self.manager: SimulationManager = None
        self.medical_state: MedicalState = None
        self.policy_props = defaultdict(bool)  # Properties that inserted/checked by the policies

    def add_to_simulation(self, manager: SimulationManager, initial_state: MedicalState):
        self.manager = manager
        self.set_medical_state_no_inform(initial_state)

    def set_test_start(self):
        self.manager.date_of_last_test[self.index] = self.manager.current_step

    def set_test_result(self, test_result):
        # TODO: add a property here
        self.manager.tested_positive_vector[self.index] = test_result
        self.manager.tested_vector[self.index] = True
        if test_result:
            self.manager.ever_tested_positive_vector[self.index] = True

    def set_medical_state_no_inform(self, new_state: MedicalState):
        self.medical_state = new_state
        self.manager.contagiousness_vector[self.index] = new_state.contagiousness[self.age]

        if new_state == self.manager.medical_machine.states_by_name["Deceased"]:
            self.manager.living_agents_vector[self.index] = False

        self.manager.susceptible_vector[self.index] = new_state.susceptible
        self.manager.test_willingness_vector[self.index] = new_state.test_willingness

    def __str__(self):
        return f"<Person,  index={self.index}, medical={self.medical_state}>"

    def get_infection_ratio(self):
        return self.medical_state.contagiousness

    def get_snapshot(self):
        geographic_circle_name = self.manager.geographic_circle_by_agent_index[self.index].name
        social_circle_snapshots = []
        for social_circle in self.manager.social_circles_by_agent_index[self.index]:
            social_circle_snapshots.append(social_circle.get_snapshot())
        return AgentSnapshot(self.index, self.age, geographic_circle_name, social_circle_snapshots)


@dataclass
class AgentSnapshot:
    index: int
    age: int
    geographic_circle: str
    social_circles: list


class AgentConstraint:
    def __init__(self, min_age, max_age, geographic_circle, social_circle_constraints):
        self.min_age = min_age
        self.max_age = max_age
        self.geographic_circle = geographic_circle
        self.social_circle_constraints = social_circle_constraints

    def meets_constraint(self, agent: AgentSnapshot):
        """

        :param agent: an AgentSnapshot of the agent you want to test against the constraint
        :return: True if the constraint is met, otherwise return False
        """
        constraint_met = True
        if not pd.isna(self.min_age) and agent.age < self.min_age:
            constraint_met = False
        if not pd.isna(self.max_age) and agent.age > self.max_age:
            constraint_met = False
        if not pd.isna(self.geographic_circle) and self.geographic_circle != agent.geographic_circle:
            constraint_met = False
        if self.social_circle_constraints is not None:
            for constraint in self.social_circle_constraints:
                if not constraint.meets_constraint(agent):
                    constraint_met = False
        return constraint_met


class SickAgents:
    def __init__(self):
        self.agent_snapshots = []

    def add_agent(self, agent_snapshot):
        self.agent_snapshots.append(agent_snapshot)

    def export(self, file_path):
        num_sick = len(self.agent_snapshots)

        export_dict = {"agent indexes": [nan] * num_sick, "geographic_circles": [nan] * num_sick,
                       "age": [nan] * num_sick}
        social_circles_num_agents = {f'{connection_type.name}_num_agents': [nan] * num_sick for connection_type in
                                     ConnectionTypes}
        social_circles_guid = {f'{connection_type.name}_guid': [nan] * num_sick for connection_type in ConnectionTypes}

        for index, agent_snapshot in enumerate(self.agent_snapshots):
            export_dict["agent indexes"][index] = agent_snapshot.index
            export_dict["geographic_circles"][index] = agent_snapshot.geographic_circle
            export_dict["age"][index] = agent_snapshot.age
            for social_circle_snapshot in agent_snapshot.social_circles:
                social_circles_num_agents[f'{social_circle_snapshot.type}_num_agents'][
                    index] = social_circle_snapshot.num_members
                social_circles_guid[f'{social_circle_snapshot.type}_guid'][index] = social_circle_snapshot.guid
        export_dict = {**export_dict, **social_circles_num_agents, **social_circles_guid}
        df_export_sick = pd.DataFrame(export_dict)
        df_export_sick.to_csv(file_path, index=False)


class InitialAgentsConstraints:
    AGE = 'age'
    GEOGRAPHIC_CIRCLE = "geographic_circles"
    RANGE_DELIMITER = '~'

    def __init__(self, constraints_file_path=None):
        self.constraints = self.parse_constraints(constraints_file_path)

    def parse_constraints(self, constraints_file_path):
        if constraints_file_path is None:
            return None
        df_constraints = pd.read_csv(constraints_file_path)
        return [self.parse_row(row) for index, row in df_constraints.iterrows()]

    def parse_row(self, row):
        min_age, max_age = self.parse_range(row[self.AGE])
        geographic_circle = row[self.GEOGRAPHIC_CIRCLE]
        social_circle_constraints = []
        for connection_type in ConnectionTypes:
            min_num, max_num = self.parse_range(row[connection_type.name])
            social_circle_constraints.append(SocialCircleConstraint(min_num, max_num, connection_type))
        return AgentConstraint(min_age, max_age, geographic_circle, social_circle_constraints)

    def parse_range(self, range_element):
        if isinstance(range_element, str):
            split_range = range_element.split(self.RANGE_DELIMITER)
            if len(split_range) == 1:
                return parse_str_to_num(split_range[0]), parse_str_to_num(split_range[0])
            if len(split_range) == 2:
                return parse_str_to_num(split_range[0]), parse_str_to_num(split_range[1])
            raise ValueError("Invalid range format!")

        else:
            return range_element, range_element
