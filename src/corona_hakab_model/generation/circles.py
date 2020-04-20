from __future__ import annotations

from typing import Set, TYPE_CHECKING

from generation.connection_types import ConnectionTypes
from dataclasses import dataclass
import pandas as pd
import uuid

if TYPE_CHECKING:
    from agent import Agent




class Circle:
    __slots__ = "kind", "agent_count"

    def __init__(self, ):
        self.agent_count = 0

    def add_many(self, agents):
        self.agent_count += len(agents)

    def remove_many(self, agents):
        self.agent_count -= len(agents)

    def add_agent(self, agent):
        self.agent_count += 1

    def remove_agent(self, agent):
        self.agent_count -= 1


class SocialCircle(Circle):
    __slots__ = ("agents", "connection_type", "guid",
                 "total_random_connections")

    def __init__(self, connection_type: ConnectionTypes):
        super().__init__()
        self.kind = "social circle"
        self.agents: Set[Agent] = set()
        self.connection_type = connection_type
        self.guid = str(uuid.uuid4())
        self.total_random_connections = 0

    def add_agent(self, agent):
        super().add_agent(agent)
        self.agents.add(agent)
        assert self.agent_count == len(self.agents)

    def remove_agent(self, agent):
        super().remove_agent(agent)
        self.agents.remove(agent)
        assert self.agent_count == len(self.agents)

    def add_many(self, agents):
        super().add_many(agents)
        self.agents.update(agents)
        assert self.agent_count == len(self.agents)

    def remove_many(self, agents):
        super().remove_many(agents)
        self.agents.difference_update(agents)
        assert self.agent_count == len(self.agents)

    def get_indexes_of_my_circle(self, my_index):
        rest_of_circle = {o.index for o in self.agents}
        rest_of_circle.remove(my_index)
        return rest_of_circle

    def get_snapshot(self):
        return SocialCircleSnapshot(self.connection_type.name, len(self.agents), self.guid)


@dataclass
class SocialCircleSnapshot:
    type: str
    num_members: int
    guid: str


class SocialCircleConstraint:
    def __init__(self, min_members, max_members, connection_type):
        self.min_members = min_members
        self.max_members = max_members
        self.connection_type = connection_type

    def meets_constraint(self, agent):
        """

        :param agent: an AgentSnapshot that we want to check if it's social circles meet the constraint
        :return: True if the constraint is trivial (min and max are nan) or if the agent contains a circle of the
        same type, which fits the constraint. Return False otherwise( agent has no such circle, or the circle doesn't
        obey the constraint
        """
        has_circle = False
        constraint_met = True
        if pd.isna(self.min_members) and pd.isna(self.max_members):
            constraint_met = True
        else:
            for social_circle in agent.social_circles:
                if social_circle.type == self.connection_type.name:
                    if social_circle.num_members < self.min_members or social_circle.num_members > self.max_members:
                        constraint_met = False
                        break
                    has_circle = True
            if not has_circle:
                constraint_met = False
        return constraint_met
