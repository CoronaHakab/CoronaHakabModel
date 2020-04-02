from agent import Agent
from typing import List, Dict
from consts import Consts
from generation_consts import GenerationConsts, GeographicalCircleDataHolder
from circles import GeographicCircle, SocialCircle
from util import dist, rv_discrete


class GenerationManager:
    __slots__ = (
        "matrix_type",
        "matrix",
        "connection_types",
        "normalize_factor",
        "total_contagious_probability",
        "agents",
        "geographic_circles",
        "social_circles",
        "generation_consts"
    )

    # todo split consts into generation_consts, simulation_consts, and plot_consts
    # todo change agent so that they wont get a simulation manager on init, and create them here
    def __init__(
            self,
            agents: List[Agent],
            generation_consts: GenerationConsts = (),
    ):
        self.generation_consts = generation_consts
        self.agents = agents

        # create geographic circles, and allocate each with agents
        self.geographic_circles: List[GeographicCircle] = []
        self.create_geographic_circles()
        self.allocate_agents()

        # set up each geographic circle
        for geo_circle in self.geographic_circles:
            geo_circle.generate_agents_ages()

    def create_geographic_circles(self):
        for geo_circle in self.generation_consts.geographic_circles:
            self.geographic_circles.append(GeographicCircle(geo_circle))

    def allocate_agents(self):
        # making sure all agents shares sum up to one. if not, normalize them
        share_factor = 1.0 / sum([geo_circle.data_holder.agents_share for geo_circle in self.geographic_circles])
        # creating a dist for selecting a geographic circle for each agent
        circle_selection = rv_discrete(values=(
            range(len(self.geographic_circles)), [geo_circle * share_factor for geo_circle in self.geographic_circles]))
        selects = iter(circle_selection.rvs(size=len(self.agents)))
        for agent in self.agents:
            selected_geo_circle = self.geographic_circles[selects.__next__()]
            selected_geo_circle.add_agent(agent)
            agent.geographic_circle = selected_geo_circle
