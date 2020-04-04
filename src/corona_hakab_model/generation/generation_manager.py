from agent import Agent
from typing import List
from generation.generation_consts import GenerationConsts
from generation.geographic_circle import GeographicCircle
from generation.connection_types import ConnectionTypes, Multi_Zone_types, Whole_Population_types
from generation.circles import SocialCircle
from util import rv_discrete


class GenerationManager:
    __slots__ = (
        "matrix_type",
        "matrix",
        "connection_types",
        "normalize_factor",
        "total_contagious_probability",
        "agents",
        "geographic_circles",
        "social_circles_by_connection_type",
        "generation_consts"
    )

    # todo split consts into generation_consts, simulation_consts, and plot_consts
    # todo change agent so that they wont get a simulation manager on init, and create them here
    def __init__(
            self,
            generation_consts: GenerationConsts = GenerationConsts(),
    ):
        self.generation_consts = generation_consts
        self.agents = [Agent(index) for index in range(self.generation_consts.population_size)]

        # create geographic circles, and allocate each with agents
        self.geographic_circles: List[GeographicCircle] = []
        self.create_geographic_circles()
        self.allocate_agents()

        # dict used to create all connection types which includes agents from multiple geographical circles
        geographic_circle_to_agents_by_connection_types = {
            connection_type: {
                circle.name: [] for circle in self.geographic_circles
            } for connection_type in Multi_Zone_types
        }

        # set up each geographic circle
        for geo_circle in self.geographic_circles:
            geo_circle.generate_agents_ages_and_connections_types()
            geo_circle.create_inner_social_circles()
            geo_circle.add_self_agents_to_dict(geographic_circle_to_agents_by_connection_types)

        # create multi-geographical social circles, and allocate agents
        for connection_type in Multi_Zone_types:
            for circle in self.geographic_circles:
                circle.create_social_circles_by_type(connection_type, geographic_circle_to_agents_by_connection_types[connection_type][circle.name])

        # fills self's social circles by connection types from all geographic circles
        self.social_circles_by_connection_type = {connection_type: [] for connection_type in ConnectionTypes}
        self.fill_social_circles()

        # create whole population circles
        self.create_whole_population_circles()

        #create_matrix_from_circles()

    def create_geographic_circles(self):
        """
        creates all geographic circles declared in generation consts.
        each circle gets an object of GeographicalCircleDataHolder
        :return:
        """
        for geo_circle in self.generation_consts.geographic_circles:
            self.geographic_circles.append(GeographicCircle(geo_circle))

    def allocate_agents(self):
        """
        splits the agents between the circles using each circle's agents share.
        :return:
        """
        # making sure all agents shares sum up to one. if not, normalize them
        share_factor = 1.0 / sum([geo_circle.data_holder.agents_share for geo_circle in self.geographic_circles])
        # creating a dist for selecting a geographic circle for each agent
        circle_selection = rv_discrete(values=(
            range(len(self.geographic_circles)), [geo_circle.data_holder.agents_share * share_factor for geo_circle in self.geographic_circles]))
        rolls = circle_selection.rvs(size=len(self.agents))
        for agent, roll in zip(self.agents, rolls):
            selected_geo_circle = self.geographic_circles[roll]
            selected_geo_circle.add_agent(agent)
            agent.geographic_circle = selected_geo_circle

    def fill_social_circles(self):
        """
        fills self social circles by connection type from self geographic circles
        :return:
        """
        for connection_type in ConnectionTypes:
            for geo_circle in self.geographic_circles:
                self.social_circles_by_connection_type[connection_type].extend(geo_circle.connection_type_to_social_circles[connection_type])

    def create_whole_population_circles(self):
        for connection_type in Whole_Population_types:
            social_circle = SocialCircle(connection_type)
            social_circle.add_many(self.agents)
            self.social_circles_by_connection_type[connection_type].append(social_circle)


gm = GenerationManager(generation_consts=GenerationConsts())
