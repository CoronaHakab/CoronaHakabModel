from typing import List

import numpy as np
from generation.circles import Circle, SocialCircle
from generation.circles_consts import GeographicalCircleDataHolder
from generation.connection_types import ConnectionTypes, In_Zone_types, Multi_Zone_types, Non_Exclusive_Types, \
    Education_Types
from util import rv_discrete


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
            for connection_type in Non_Exclusive_Types:
                if np.random.random() < self.data_holder.connection_types_prob_by_age[age][connection_type]:
                    agent_connection_types.append(connection_type)
            
            if agent.is_adult() and np.random.random() < self.data_holder.connection_types_prob_by_age[age][
                ConnectionTypes.Work]:

                # if the agent works, decide workplace
                agent_connection_types.append(workplace_distribution.rvs())

            education_prob = [self.data_holder.connection_types_prob_by_age[age][connection_type] for
                              connection_type in Education_Types]
            education_type = rv_discrete(values=([Education_Types + [-1], education_prob + [1 - sum(education_prob)]])).rvs()

            if not agent.is_adult() and education_type > 0:
                agent_connection_types.append(education_type)

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
        # calculate amount of agents for each size group
        # we'll also use size_num_agents to count how many agents were placed in each size group.
        possible_sizes, probs = self.data_holder.circles_size_distribution_by_connection_type[connection_type]
        size_num_agents = {size : 0 for size in possible_sizes}             
        rolls = np.random.choice(possible_sizes, size=len(agents_for_type), p=probs)
        for roll in rolls:
            size_num_agents[roll] += 1


        # populate circles in each size group
        for size in possible_sizes:
            # create circles
            amount_of_circles = max(1, round(size_num_agents[size] / size))
            circles = [SocialCircle(connection_type) for _ in range(amount_of_circles)]
            # index is used to go over all circles in the size group s.t. the population is divided as qeually as possible
            index = 0
            
            # if the distribution is age dependent, fill adults first.
            # check if there is a distribution of adults in for the connection_type
            adult_type_distribution = self.data_holder.adult_distributions.get(connection_type)
            if adult_type_distribution:
                # get random amount of adults for each circle according to distribution
                circles_adult_number = [adult_type_distribution[size].rvs() for _ in range(amount_of_circles)]
                # devide population according to age
                adults = [agent for agent in agents_for_type if agent.age > 18]
                non_adults = [agent for agent in agents_for_type if agent.age <= 18]
                # place adults where needed, non adults elsewhere, where circles need to be populated
                while len(adults) > 0 and size_num_agents[size] > 0:
                    circle = circles[index % len(circles)]
                    agent = None
                    # if there are more adults needed, or no more kids, circle gets an adult
                    if circles_adult_number[index % len(circles)] > circle.agent_count or len(non_adults) == 0:
                        agent = adults.pop()
                    else:
                        agent = non_adults.pop()
                    circle.add_agent(agent)
                    agents_for_type.remove(agent)
                    index += 1
                    size_num_agents[size] -= 1
           
            # fill in the rest of the population 
            while len(agents_for_type) > 0 and size_num_agents[size] > 0:
                agent = agents_for_type.pop()
                circle = circles[index % len(circles)]
                circle.add_agent(agent)
                index += 1
                size_num_agents[size] -= 1
            
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
