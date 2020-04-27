from typing import List

import numpy as np
from generation.circles import Circle, SocialCircle
from generation.circles_consts import GeographicalCircleDataHolder
from generation.connection_types import ConnectionTypes, In_Zone_types, Multi_Zone_types, Education_Types, \
    Non_Random_Age_Types
from util import rv_discrete, lower_bound


class GeographicCircle(Circle):
    __slots__ = (
        "name",
        "agents",
        "all_social_circles",
        "data_holder",
        "circle_count",
        "connection_type_to_agents",
        "connection_type_to_social_circles",

    )

    def __init__(self, data_holder: GeographicalCircleDataHolder):
        super().__init__()
        self.circle_count = 0
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
        It allows choosing whether some one goes to work, to school, to kindergarten or to none of those.
        :return:
        """
        ages = iter(self.data_holder.age_distribution.rvs(size=len(self.agents)))
        workplace_distribution = rv_discrete(
            values=([ConnectionTypes.School, ConnectionTypes.Kindergarten, ConnectionTypes.Work],
                    [self.data_holder.teachers_workforce_ratio, self.data_holder.kindergarten_workforce_ratio,
                    1 - self.data_holder.teachers_workforce_ratio - self.data_holder.kindergarten_workforce_ratio])
        )
        for agent, age in zip(self.agents, ages):
            agent.age = age
            agent_connection_types = []
            
            if agent.is_adult() and np.random.random() < self.data_holder.connection_types_prob_by_age[age][
                ConnectionTypes.Work]:

                # if the agent works, decide workplace
                agent_connection_types.append(workplace_distribution.rvs())

            if not agent.is_adult():
                education_prob = [self.data_holder.connection_types_prob_by_age[age][connection_type] for
                                  connection_type in Education_Types]
                if np.random.random() < sum(education_prob):
                    # normalize the probabilities
                    education_type = rv_discrete(
                        values=(Education_Types, [prob / sum(education_prob) for prob in education_prob])).rvs()
                    agent_connection_types.append(education_type)

            # handle other connection types. This single FOR condition contains words "connection" and "type" 9 times
            for connection_type in [connection_type for connection_type in ConnectionTypes if connection_type not in [
                ConnectionTypes.School, ConnectionTypes.Work, ConnectionTypes.Kindergarten]]:

                if np.random.random() < self.data_holder.connection_types_prob_by_age[age][connection_type]:
                    agent_connection_types.append(connection_type)

            for connection_type in agent_connection_types:
                self.connection_type_to_agents[connection_type].append(agent)
            
    def create_inner_social_circles(self):
        # todo notice that family connection types doesnt notice between ages
        for connection_type in In_Zone_types:
            self.create_social_circles_by_type(connection_type, self.connection_type_to_agents[connection_type])

    def create_social_circles_by_type(self, connection_type: ConnectionTypes, agents_for_type: List["Agent"]):
        """
        creates social circles of a given connection type, with a given list of agents.
        uses self data holder circle size distribution of the given connection type.
        NOTE: last circle might run out of adults if the given agents weren't created with enough adults
        :param connection_type: the connection type currently created
        :param agents_for_type: the agents that will be inserted to the social circles
        :return:
        """

        np.random.shuffle(agents_for_type)
        possible_sizes, probs = self.data_holder.circles_size_distribution_by_connection_type[connection_type]
        if len(possible_sizes) == 0 and len(probs) == 0:
            return

        circles = []

        while len(agents_for_type) > 0:
            circle_size = np.random.choice(possible_sizes, p=probs)

            # if not enough agents, or next circle would be to small, create circle of abnormal size
            if len(agents_for_type) < circle_size + min(possible_sizes):
                circle_size = len(agents_for_type)

            # if the distribution is age dependent, fill with the appropriate age proportions
            if connection_type in Non_Random_Age_Types:
                circles.append(self.create_age_dependant_circle(connection_type, agents_for_type, circle_size))

            else:
                circle = SocialCircle(connection_type)
                for _ in range(circle_size):
                    agent = agents_for_type.pop()
                    assert agent not in circle.agents
                    circle.add_agent(agent)
                circles.append(circle)

        self.connection_type_to_social_circles[connection_type].extend(circles)
        self.all_social_circles.extend(circles)

    def create_age_dependant_circle(self, connection_type, agents_for_type, size):
        circle = SocialCircle(connection_type)

        # if circle size is abnormal, number of adults is taken from the closest smaller possible size
        possible_sizes, _ = self.data_holder.circles_size_distribution_by_connection_type[connection_type]
        if size in possible_sizes:
            adult_type_distribution = self.data_holder.adult_distributions.get(connection_type)[size]
        else:
            approx_size = min([psize for psize in possible_sizes if psize < size], key=lambda el: abs(el - size))
            adult_type_distribution = self.data_holder.adult_distributions.get(connection_type)[approx_size]

        adults = [agent for agent in agents_for_type if agent.age > 18]
        non_adults = [agent for agent in agents_for_type if agent.age <= 18]

        adult_num = min(round(adult_type_distribution.rvs()), len(adults))
        child_num = min(size - adult_num, len(non_adults))
        for _ in range(adult_num):
            agent = adults.pop()
            assert agent not in circle.agents
            circle.add_agent(agent)
            agents_for_type.remove(agent)

        for _ in range(child_num):
            agent = non_adults.pop()
            assert agent not in circle.agents
            circle.add_agent(agent)
            agents_for_type.remove(agent)

        # if there is place left in the circle, fill it with agents:
        if circle.agent_count < size:
            agents = agents_for_type[: size - circle.agent_count]
            del agents_for_type[: size - circle.agent_count]
            circle.add_many(agents)

        return circle

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
