import pickle
from matplotlib import pyplot as plt
import numpy as np
import math

from generation.circles_generator import PopulationData
from generation.matrix_generator import MatrixData
from generation.connection_types import ConnectionTypes


class PopulationAnylzer:
    """
    this module is in charge of plotting statistics about the generation stage, and specifically the population
    """
    __slots__ = "population_data", "matrix_data"

    def __init__(self, population_data_path, matrix_data_path=None):

        # importing population data from path
        try:
            with open(population_data_path, "rb") as population_data:
                self.population_data: PopulationData = pickle.load(population_data)
        except FileNotFoundError:
            raise FileNotFoundError("population analyzer couldn't open population data")

        # importing matrix data from path.
        # for now, allows not importing a matrix data
        if matrix_data_path is not None:
            try:
                with open(matrix_data_path, "rb") as matrix_data:
                    self.matrix_data: MatrixData = pickle.load(matrix_data)
            except FileNotFoundError:
                raise FileNotFoundError("population analyzer couldn't open matrix data")

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
            size_count = {}
            for circle in circles:
                size = circle.agent_count
                if size not in size_count:
                    size_count[size] = 0
                size_count[size] += 1
            # creating a bar graph with a bar for each circle size
            bars = [size for size in sorted(size_count.keys())]
            heights = [size_count[size] for size in bars]
            x = np.arange(len(bars))
            ax.bar(x, heights)
            ax.set_xticks(x)
            ax.set_xticklabels(bars)
        plt.show()

    def plot_agents_ages(self):
        agents = self.population_data.agents
        ages = [agent.age for agent in agents]
        max_age = max(ages)     # calculated for bins
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
