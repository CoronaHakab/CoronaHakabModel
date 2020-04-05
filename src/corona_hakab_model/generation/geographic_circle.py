from typing import List

import numpy as np
from generation.circles import Circle, SocialCircle
from generation.circles_consts import GeographicalCircleDataHolder
from generation.connection_types import ConnectionTypes, In_Zone_types, Multi_Zone_types
from util import dist, rv_discrete


class GeographicCircle(Circle):
    __slots__ = (
        "name",
        "agents",
        "all_social_circles",
        "data_holder",
        "connection_type_to_agents",
        "connection_type_to_social_circles",
    )

    def __init__(self, data_holder: GeographicalCircleDataHolder):
        super().__init__()
        self.kind = "geographic circle"
        self.agents = []
        self.data_holder = data_holder
        self.connection_type_to_agents = {con_type: [] for con_type in ConnectionTypes}
        self.connection_type_to_social_circles = {con_type: [] for con_type in ConnectionTypes}
        self.all_social_circles = []
        self.name = data_holder.name

    def generate_agents_ages_and_connections_types(self):
        """
        Iterates over each agent. Generates it's age by a given age distribution (self.data_holder.age_distribution).
        For each agent, iterates over each connection type, and rolls whether it has this type of connection or not.
        Gets this probabilities from self.data_holder.connection_types_prob_by_age.
        It allows choosing whether some one goes to work, to school, or to none of those.
        :return:
        """
        ages = iter(self.data_holder.age_distribution.rvs(size=len(self.agents)))
        for agent, age in zip(self.agents, ages):
            agent.age = age
            for connection_type in ConnectionTypes:
                if np.random.random() < self.data_holder.connection_types_prob_by_age[age][connection_type]:
                    self.connection_type_to_agents[connection_type].append(agent)

    def create_inner_social_circles(self):
        # todo notice that family connection types doesnt notice between ages
        for connection_type in In_Zone_types:
            self.create_social_circles_by_type(connection_type, self.connection_type_to_agents[connection_type])

    def create_social_circles_by_type(self, connection_type: ConnectionTypes, agents_for_type: List["Agent"]):
        """
        creates social circles of a given connection type, with a given list of agents.
        uses self data holder circle size distribution of the given connection type
        :param connection_type: the connection type currently created
        :param agents_for_type: the agents that will be inserted to the social circles
        :return:
        """
        possible_sizes, probs = self.data_holder.circles_size_distribution_by_connection_type[connection_type]
        size_to_agents = {size: [] for size in possible_sizes}

        np.random.shuffle(agents_for_type)
        rolls = iter(np.random.choice(possible_sizes, size=len(agents_for_type), p=probs))

        for agent, roll in zip(agents_for_type, rolls):
            size_to_agents[roll].append(agent)

        for size, agents in size_to_agents.items():
            amount_of_circles = max(1, round(len(agents) / size))
            circles = [SocialCircle(connection_type) for _ in range(amount_of_circles)]
            # doing it this way so that all circles will be the same size
            index = 0
            while len(agents) > 0:
                agent = agents.pop()
                circle = circles[index % len(circles)]
                circle.add_agent(agent)
                agent.social_circles.append(circle)
                index += 1
            self.connection_type_to_social_circles[connection_type].extend(circles)
            self.all_social_circles.extend(circles)

    def add_self_agents_to_dict(self, geographic_circle_to_agents_by_connection_types):
        for connection_type in Multi_Zone_types:
            agents = self.connection_type_to_agents[connection_type]
            circles_names = list(
                self.data_holder.multi_zone_connection_type_to_geo_circle_probability[connection_type].keys()
            )
            circles_probabilites = list(
                self.data_holder.multi_zone_connection_type_to_geo_circle_probability[connection_type].values()
            )
            rolls = np.random.choice(circles_names, size=len(agents), p=circles_probabilites)
            for agent, roll in zip(agents, rolls):
                geographic_circle_to_agents_by_connection_types[connection_type][roll].append(agent)

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
