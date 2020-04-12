import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from generation.circles_generator import PopulationData
from generation.connection_types import ConnectionTypes

OUTPUT_DIR = Path("../../../output/")


class PopulationAnalyzer:
    """
    this module is in charge of plotting statistics about the population
    """

    __slots__ = "population_data"
    circle_size_data = {}

    def __init__(self, population_data_path):

        # importing population data from path
        self.population_data = PopulationData.import_population_data(population_data_path)

        for conn_type in ConnectionTypes:
            circles = self.population_data.social_circles_by_connection_type[conn_type]
            self.circle_size_data[conn_type] = defaultdict(int)
            for circle in circles:
                self.circle_size_data[conn_type][circle.agent_count] += 1

    def plot_circles_sizes(self):
        """
        plotting a bar chart for each connection type, showing the circle sizes count.
        """
        fig = plt.figure()
        types_amount = len(ConnectionTypes)
        grid_count = math.ceil(math.sqrt(types_amount))
        for i, con_type in enumerate(ConnectionTypes, 1):
            ax = fig.add_subplot(grid_count, grid_count, i)
            ax.set_title(str(con_type))
            ax.set_ylabel("number of circles")
            ax.set_xlabel("circle size")
            # creating a bar graph with a bar for each circle size
            bars, heights = zip(*sorted(self.circle_size_data[con_type].items()))
            x = np.arange(len(bars))
            ax.bar(x, heights)
            ax.set_xticks(x)
            ax.set_xticklabels(bars)
        plt.show()

    def plot_agents_ages(self):
        agents = self.population_data.agents
        ages = [agent.age for agent in agents]
        max_age = max(ages)  # calculated for bins
        plt.hist(ages, bins=max_age, range=(0.5, max_age + 0.5))
        plt.gca().set_title("agent age histogram")
        plt.gca().set_xlabel("age")
        plt.gca().set_ylabel("count")
        plt.show()

    def export_csv(self):
        connection_type_circle_size_data = [(conn_type.name, circle_agent_count, number_of_circles)
                                            for conn_type in ConnectionTypes
                                            for (circle_agent_count,
                                                 number_of_circles) in self.circle_size_data[conn_type].items()]
        connection_type_circle_size_df = pd.DataFrame(connection_type_circle_size_data,
                                                      columns=["connection_type", "circle_size", "number_of_circles"])
        connection_type_circle_size_df.to_csv(OUTPUT_DIR / "connection_type_circle_size_histogram_data.csv")

        agent_age_bins: np.ndarray = np.bincount([agent.age for agent in self.population_data.agents])
        agent_age_histogram_data = list(zip(np.flatnonzero(agent_age_bins), agent_age_bins[np.nonzero(agent_age_bins)]))

        agent_age_histogram_df = pd.DataFrame(agent_age_histogram_data, columns=["age", "count"])
        agent_age_histogram_df.to_csv(OUTPUT_DIR / "agent_age_histogram.csv")


if __name__ == "__main__":
    file_path = OUTPUT_DIR / "population_data.pickle"
    population_analyzer = PopulationAnalyzer(file_path)
    population_analyzer.plot_circles_sizes()
    population_analyzer.plot_agents_ages()
    population_analyzer.export_csv()
