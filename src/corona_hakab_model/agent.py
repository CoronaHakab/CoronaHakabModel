from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from generation.connection_types import ConnectionTypes

if TYPE_CHECKING:
    from medical_state import MedicalState
    from manager import SimulationManager


class Agent:
    """
    This class represents a person in our doomed world.
    """

    __slots__ = ("index", "medical_state", "manager", "age")

    # todo note that this changed to fit generation. should update simulation manager accordingly
    def __init__(self, index):
        self.index = index

        # don't know if this is necessary
        self.manager: SimulationManager = None
        self.medical_state: MedicalState = None


    # def set_medical_state_no_inform(self, new_state: MedicalState):
    #     self.medical_state = new_state
    #     self.manager.contagiousness_vector[self.index] = new_state.contagiousness
    #
    #     if new_state == self.manager.medical_machine.states_by_name["Deceased"]:
    #         self.manager.living_agents_vector[self.index] = False
    #
    #     self.manager.susceptible_vector[self.index] = new_state.susceptible
    #     self.manager.test_willingness_vector[self.index] = new_state.test_willingness

    def __str__(self):
        return f"<Person,  index={self.index}, medical={self.medical_state}>"

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


class InitialSickAgents:
    EXPORT_OUTPUT_DIR = "../../output/"
    EXPORT_FILE_NAME = "initial_sick.csv"

    def __init__(self):
        self.agent_snapshots = []

    def add_agent(self, agent_snapshot):
        self.agent_snapshots.append(agent_snapshot)

    def add_many_agents(self, agents: pd.DataFrame):
        self.agent_snapshots.append(agents)

    def export(self):
        return  # FIXME later
        num_sick = len(self.agent_snapshots)
        export_dict = {"agent indexes": [0] * num_sick, "geographic_circles": [0] * num_sick, "age": [0] * num_sick}
        social_circles = {connection_type.name: [0] * num_sick for connection_type in ConnectionTypes}
        for index, agent_snapshot in enumerate(self.agent_snapshots):
            export_dict["agent indexes"][index] = agent_snapshot.index
            export_dict["geographic_circles"][index] = agent_snapshot.geographic_circle
            export_dict["age"][index] = agent_snapshot.age
            for social_circle_snapshot in agent_snapshot.social_circles:
                social_circles[social_circle_snapshot.type][index] = social_circle_snapshot.num_members
        export_dict = {**export_dict, **social_circles}
        df_export_sick = pd.DataFrame(export_dict)
        df_export_sick.to_csv(self.EXPORT_OUTPUT_DIR + self.EXPORT_FILE_NAME, index=False)


class Circle:
    __slots__ = "kind", "agent_count"

    def __init__(self):
        self.agent_count = 0

    def add_many(self, agents):
        self.agent_count += len(agents)

    def remove_many(self, agents):
        self.agent_count -= len(agents)

    def add_agent(self, agent):
        self.agent_count += 1

    def remove_agent(self, agent):
        self.agent_count -= 1


class TrackingCircle(Circle):
    __slots__ = ("agents", "ever_visited")

    def __init__(self):
        super().__init__()
        self.agents = set()
        # done to count how many different agents ever visited a given state
        self.ever_visited = set()

    def add_agent(self, agent):
        super().add_agent(agent)
        if agent in self.agents:
            raise ValueError("DuplicateAgent")
        self.agents.add(agent)
        self.ever_visited.add(agent)
        assert self.agent_count == len(self.agents)

    def remove_agent(self, agent):
        super().remove_agent(agent)
        self.agents.remove(agent)
        assert self.agent_count == len(self.agents)

    def add_many(self, agents):
        super().add_many(agents)
        if self.agents.intersection(set(agents)):
            raise ValueError("DuplicateAgent")
        self.agents.update(agents)
        self.ever_visited.update(agents)
        assert self.agent_count == len(
            self.agents
        ), f"self.agent_count: {self.agent_count}, len(self.agents): {len(self.agents)}"

    def remove_many(self, agents):
        super().remove_many(agents)
        self.agents.difference_update(agents)
        assert self.agent_count == len(self.agents)

    def get_indexes_of_my_circle(self, my_index):
        rest_of_circle = {o.index for o in self.agents}
        rest_of_circle.remove(my_index)
        return rest_of_circle
