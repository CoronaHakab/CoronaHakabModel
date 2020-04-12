import math
from collections import defaultdict

import numpy as np
from generation.circles_generator import PopulationData
from generation.connection_types import ConnectionTypes
from matplotlib import pyplot as plt


class PopulationAnylzer:
    """
    this module is in charge of plotting statistics about the population
    """

    __slots__ = "population_data"

    def __init__(self, population_data_path):

        # importing population data from path
        self.population_data = PopulationData.import_population_data(population_data_path)

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
            ax.set_ylabel("amount of circles")
            ax.set_xlabel("circle size")
            circles = self.population_data.social_circles_by_connection_type[con_type]
            size_count = defaultdict(int)
            for circle in circles:
                size_count[circle.agent_count] += 1
            # creating a bar graph with a bar for each circle size
            bars, heights = zip(*sorted(size_count.items()))
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
        plt.title = "agents ages histogram"
        plt.xlabel = "age"
        plt.ylabel = "count"
        plt.show()


if __name__ == "__main__":
    file_path = "../../output/population_data.pickle"
    population_analyzer = PopulationAnylzer(file_path)
    population_analyzer.plot_circles_sizes()
    population_analyzer.plot_agents_ages()
