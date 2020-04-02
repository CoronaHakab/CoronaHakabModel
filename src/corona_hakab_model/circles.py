from agent import Agent
from typing import List, Dict, DefaultDict
from util import dist, rv_discrete
from consts import Consts
from generation_consts import GeographicalCircleDataHolder, GenerationConsts


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


class GeographicCircle(Circle):
    __slots__ = (
        "agents",
        "social_circles",
        "data_holder",
    )

    def __init__(self, data_holder: GeographicalCircleDataHolder):
        super().__init__()
        self.agents = []
        self.data_holder = data_holder

    def generate_agents_ages(self):
        ages = iter(self.data_holder.age_distribution.rvs(size=len(self.agents)))
        for agent in self.agents:
            agent.age = ages.__next__()

    def add_agent(self, agent):
        super().add_agent(agent)
        self.agents.append(agent)
        assert self.agent_count == len(self.agents)

    def remove_agent(self, agent):
        super().remove_agent(agent)
        self.agents.remove(agent)
        assert self.agent_count == len(self.agents)

    def add_many(self, agents):
        super().add_many(agents)
        self.agents.extend(agents)
        assert self.agent_count == len(self.agents)

    def remove_many(self, agents):
        super().remove_many(agents)
        self.agents.remove(agents)
        assert self.agent_count == len(self.agents)


class SocialCircle(Circle):
    __slots__ = ("agents",)

    def __init__(self):
        super().__init__()
        self.agents = set()

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