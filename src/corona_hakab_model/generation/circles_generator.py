import pickle
import sys
from typing import List, Dict
import os.path
from project_structure import OUTPUT_FOLDER

import numpy as np

from common.agent import Agent
from __data__ import __version__
from generation import connection_types
from common.social_circle import SocialCircle
from generation.circles_consts import CirclesConsts
from generation.connection_types import ConnectionTypes, Multi_Zone_types, Whole_Population_types
from generation.geographic_circle import GeographicCircle

# for exporting with pickle (/serializing) - set a high recursion rate
sys.setrecursionlimit(5000)


class PopulationData:
    __slots__ = (
        "version",
        "agents",
        "geographic_circles",
        "social_circles_by_connection_type",
        "geographic_circle_by_agent_index",
        "social_circles_by_agent_index",
        "num_of_random_connections",
        "random_connections_strength"
    )

    def __init__(self):
        self.version = __version__
        self.agents = []
        self.geographic_circles: List[GeographicCircle] = []
        self.social_circles_by_connection_type: Dict[ConnectionTypes, List[SocialCircle]] = {}
        self.geographic_circle_by_agent_index = {}
        self.social_circles_by_agent_index = {}
        self.num_of_random_connections: np.ndarray = np.array([])
        self.random_connections_strength: np.ndarray = np.array([])

    def export(self, export_path, file_name: str):
        if not file_name.endswith(".pickle"):
            file_name += ".pickle"

        with open(os.path.join(export_path, file_name), "wb") as export_file:
            pickle.dump(self, export_file)

    @staticmethod
    def import_population_data(import_file_path: str) -> "PopulationData":
        with open(import_file_path, "rb") as import_file:
            population_data = pickle.load(import_file)

        # pickle's version should be updated per application's version
        assert population_data.version == __version__
        return population_data


class CirclesGenerator:
    # todo organize all path fields in a single file
    # import/export variables
    EXPORT_OUTPUT_DIR = OUTPUT_FOLDER
    EXPORT_FILE_NAME = "population_data.pickle"

    # todo split consts into generation_consts, simulation_consts, and plot_consts

    def __init__(
            self, circles_consts: CirclesConsts,
    ):
        self.circles_consts = circles_consts
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

        # fills self's geographic circle by agent from all geographic circles
        self.geographic_circle_by_agent_index = {}
        self.fill_geographic_circle_by_agent_index()

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
        self.social_circles_by_connection_type: Dict[ConnectionTypes, List[SocialCircle]] =\
            {connection_type: [] for connection_type in ConnectionTypes}
        self.fill_social_circles()

        # create whole population circles
        self.create_whole_population_circles()

        # fills self's social circles by agent index from all social circles
        self.social_circles_by_agent_index = {}
        self.fill_social_circles_by_agent_index()

        # Random connections in circles TODO: Move to matrix generation
        self.num_of_random_connections = np.zeros((len(self.agents), len(ConnectionTypes)), dtype=float)
        self.random_connections_strength = np.zeros(len(ConnectionTypes), dtype=float)
        self.generate_random_connections()

        self._fill_population_data()

    def generate_random_connections(self):
        for connection_type in connection_types.With_Random_Connections + connection_types.With_Geo_Random_Connections:
            exp_mean = self.circles_consts.random_connections_dist_mean[connection_type]

            if connection_type in self.circles_consts.random_connections_strength_factor:
                self.random_connections_strength[connection_type] = self.circles_consts.random_connections_strength_factor[connection_type]

            for circle in self.social_circles_by_connection_type[connection_type]:
                num_of_agents_in_circle = len(circle.agents)
                if num_of_agents_in_circle == 1:
                    # No random connections
                    continue

                agents_id = [a.index for a in circle.agents]

                # Sample from exponential distribution
                rand_connections = np.random.exponential(exp_mean, len(agents_id))

                # You can't have more random connections than the number of people (other than you) in the circle
                rand_connections = np.clip(rand_connections, 0, num_of_agents_in_circle - 1)
                circle.total_random_connections = np.sum(rand_connections)

                self.num_of_random_connections[agents_id, [connection_type] * len(agents_id)] = rand_connections

    def create_geographic_circles(self):
        """
        creates all geographic circles declared in generation consts.
        each circle gets an object of GeographicalCircleDataHolder
        :return:
        """
        for geo_circle in self.circles_consts.get_geographic_circles():
            self.geographic_circles.append(GeographicCircle(geo_circle))

    def allocate_agents(self):
        """
        splits the agents between the circles using each circle's agents share.
        :return:
        """
        # making sure all agents shares sum up to one. if not, normalize them
        share_factor = 1.0 / sum([geo_circle.data_holder.agents_share for geo_circle in self.geographic_circles])
        # creating a dist for selecting a geographic circle for each agent
        rolls = np.random.choice(
            np.arange(len(self.geographic_circles)), size=len(self.agents),
            p=[geo_circle.data_holder.agents_share * share_factor for geo_circle in self.geographic_circles]
        )
        for agent, roll in zip(self.agents, rolls):
            selected_geo_circle = self.geographic_circles[roll]
            selected_geo_circle.add_agent(agent)

    def fill_geographic_circle_by_agent_index(self):
        """
        fills self geographic circle by agent index from self geographic circles
        :return:
        """
        for geo_circle in self.geographic_circles:
            for agent in geo_circle.agents:
                self.geographic_circle_by_agent_index[agent.index] = geo_circle

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
            agents = []
            for geo_circle in self.geographic_circles:
                agents.extend(geo_circle.connection_type_to_agents[connection_type])
            social_circle.add_many(agents)
            self.social_circles_by_connection_type[connection_type].append(social_circle)

    def fill_social_circles_by_agent_index(self):
        """
        fills self social circles by agent index from self social circles
        :return:
        """
        for social_circles in self.social_circles_by_connection_type.values():
            for social_circle in social_circles:
                for agent in social_circle.agents:
                    self.social_circles_by_agent_index.setdefault(agent.index, []).append(social_circle)

    def _fill_population_data(self):
        # fill population data with my data
        self.population_data.agents = self.agents
        self.population_data.social_circles_by_connection_type = self.social_circles_by_connection_type
        self.population_data.geographic_circles = self.geographic_circles
        self.population_data.geographic_circle_by_agent_index = self.geographic_circle_by_agent_index
        self.population_data.social_circles_by_agent_index = self.social_circles_by_agent_index
        self.population_data.num_of_random_connections = self.num_of_random_connections
        self.population_data.random_connections_strength = self.random_connections_strength

    def export(self):
        # export population data using pickle
        self.population_data.export(self.EXPORT_OUTPUT_DIR, self.EXPORT_FILE_NAME)

    def import_population_data(self, import_file_path=None):
        if import_file_path is None:
            import_file_path = self.EXPORT_OUTPUT_DIR + self.EXPORT_FILE_NAME

        self.population_data = PopulationData.import_population_data(import_file_path)

        # pickle's version should be updated per application's version
        assert self.population_data.version == __version__

        # fill imported data to self
        self.agents = self.population_data.agents
        self.social_circles_by_connection_type = self.population_data.social_circles_by_connection_type
        self.geographic_circles = self.population_data.geographic_circles
        self.geographic_circle_by_agent_index = self.population_data.geographic_circle_by_agent_index
        self.social_circles_by_agent_index = self.population_data.social_circles_by_agent_index
