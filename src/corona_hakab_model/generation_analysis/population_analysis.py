import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import project_structure
from matplotlib import pyplot as plt
import argparse

from generation.circles_generator import PopulationData
from generation.connection_types import ConnectionTypes


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

    def export_csv(self, circle_file_path: str, age_file_path: str):
        connection_type_circle_size_data = [(conn_type.name, circle_agent_count, number_of_circles)
                                            for conn_type in ConnectionTypes
                                            for (circle_agent_count,
                                                 number_of_circles) in self.circle_size_data[conn_type].items()]
        connection_type_circle_size_df = pd.DataFrame(connection_type_circle_size_data,
                                                      columns=["connection_type", "circle_size", "number_of_circles"])
        connection_type_circle_size_df.to_csv(circle_file_path)

        agent_age_bins: np.ndarray = np.bincount([agent.age for agent in self.population_data.agents])
        agent_age_histogram_data = list(zip(np.flatnonzero(agent_age_bins), agent_age_bins[np.nonzero(agent_age_bins)]))

        agent_age_histogram_df = pd.DataFrame(agent_age_histogram_data, columns=["age", "count"])
        agent_age_histogram_df.to_csv(age_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Population Analyser")
    parser.add_argument("-d", "--dir", dest='directory', type=str, default='',
                        help='Directory containing population data to analyse, and in which to save results.')
    parser.add_argument("-p", "--population", dest="population", type=str, default='',
                        help='Path to population data pickle. Prioritized over directory if both given.')
    parser.add_argument("-a", "--age", dest="age", type=str, default='',
                        help="Path to resulting age histogram. Prioritized over directory if both given")
    parser.add_argument("-c", "--circle", dest="circle", type=str, default='',
                        help="Path to resulting circle histogram. Prioritized over directory if both given")
    parser.add_argument("--no-view",
                        dest='view',
                        default=True,
                        action='store_false',
                        help="Skip viewing the generated histograms.")
    parser.add_argument("--no-save",
                        dest='save',
                        default=True,
                        action='store_false',
                        help="Skip saving the resulting CSV")
    args = parser.parse_args()
    directory = Path(args.directory or project_structure.OUTPUT_FOLDER)
    pop_file_path = args.population or directory / "population_data.pickle"
    age_file_path = args.age or directory / "agent_age_histogram.csv"
    circle_file_path = args.circle or directory / "connection_type_circle_size_histogram_data.csv"
    population_analyzer = PopulationAnalyzer(pop_file_path)
    if args.view:
        population_analyzer.plot_circles_sizes()
        population_analyzer.plot_agents_ages()
    if args.save:
        population_analyzer.export_csv(circle_file_path, age_file_path)
