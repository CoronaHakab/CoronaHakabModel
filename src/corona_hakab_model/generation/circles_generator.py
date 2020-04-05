# for exporting with pickle (/serializing) - set a high recursion rate
import pickle
import sys
from typing import List

from agent import Agent
from generation.circles import SocialCircle
from generation.circles_consts import CirclesConsts
from generation.connection_types import ConnectionTypes, Multi_Zone_types, Whole_Population_types
from generation.geographic_circle import GeographicCircle
from util import rv_discrete

sys.setrecursionlimit(5000)


class PopulationData:
    __slots__ = ("agents", "geographic_circles", "social_circles_by_connection_type")

    def __init__(self):
        self.agents = []
        self.geographic_circles: List[GeographicCircle] = []
        self.social_circles_by_connection_type = {}


class CirclesGenerator:

    # import/export variables
    EXPORT_OUTPUT_DIR = "../../output/"
    EXPORT_FILE_NAME = "population_data.pickle"

    # todo split consts into generation_consts, simulation_consts, and plot_consts
    def __init__(
        self, generation_consts: CirclesConsts = CirclesConsts(),
    ):
        self.circles_consts = generation_consts
        self.population_data = PopulationData()
        self.agents = [Agent(index) for index in range(self.circles_consts.population_size)]

        # create geographic circles, and allocate each with agents
        self.geographic_circles: List[GeographicCircle] = []
        self.create_geographic_circles()
        self.allocate_agents()

        # dict used to create all connection types which includes agents from multiple geographical circles
        geographic_circle_to_agents_by_connection_types = {
            connection_type: {circle.name: [] for circle in self.geographic_circles}
            for connection_type in Multi_Zone_types
        }

        # set up each geographic circle
        for geo_circle in self.geographic_circles:
            geo_circle.generate_agents_ages_and_connections_types()
            geo_circle.create_inner_social_circles()
            geo_circle.add_self_agents_to_dict(geographic_circle_to_agents_by_connection_types)

        # create multi-geographical social circles, and allocate agents
        for connection_type in Multi_Zone_types:
            for circle in self.geographic_circles:
                circle.create_social_circles_by_type(
                    connection_type, geographic_circle_to_agents_by_connection_types[connection_type][circle.name]
                )

        # fills self's social circles by connection types from all geographic circles
        self.social_circles_by_connection_type = {connection_type: [] for connection_type in ConnectionTypes}
        self.fill_social_circles()

        # create whole population circles
        self.create_whole_population_circles()

        # export the population data
        self.export()

    def create_geographic_circles(self):
        """
        creates all geographic circles declared in generation consts.
        each circle gets an object of GeographicalCircleDataHolder
        :return:
        """
        for geo_circle in self.circles_consts.geographic_circles:
            self.geographic_circles.append(GeographicCircle(geo_circle))

    def allocate_agents(self):
        """
        splits the agents between the circles using each circle's agents share.
        :return:
        """
        # making sure all agents shares sum up to one. if not, normalize them
        share_factor = 1.0 / sum([geo_circle.data_holder.agents_share for geo_circle in self.geographic_circles])
        # creating a dist for selecting a geographic circle for each agent
        circle_selection = rv_discrete(
            values=(
                range(len(self.geographic_circles)),
                [geo_circle.data_holder.agents_share * share_factor for geo_circle in self.geographic_circles],
            )
        )
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
                self.social_circles_by_connection_type[connection_type].extend(
                    geo_circle.connection_type_to_social_circles[connection_type]
                )

    def create_whole_population_circles(self):
        for connection_type in Whole_Population_types:
            social_circle = SocialCircle(connection_type)
            social_circle.add_many(self.agents)
            self.social_circles_by_connection_type[connection_type].append(social_circle)

    def export(self):
        # fill population data with my data
        self.population_data.agents = self.agents
        self.population_data.social_circles_by_connection_type = self.social_circles_by_connection_type
        self.population_data.geographic_circles = self.geographic_circles

        # export population data using pickle
        # todo fix recursion error
        # with open(self.EXPORT_OUTPUT_DIR + self.EXPORT_FILE_NAME, 'wb') as export_file:
        #   pickle.dump(self.population_data, export_file)

    def import_population_data(self, import_file_path=None):
        if import_file_path is None:
            import_file_path = self.EXPORT_OUTPUT_DIR + self.EXPORT_FILE_NAME

        with open(import_file_path, "rb") as import_file:
            self.population_data = pickle.load(import_file)

        # fill imported data to self
        self.agents = self.population_data.agents
        self.social_circles_by_connection_type = self.population_data.social_circles_by_connection_type
        self.geographic_circles = self.population_data.geographic_circles