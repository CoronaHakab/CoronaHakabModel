from pathlib import Path
from generation.circles_generator import PopulationData
from generation.connection_types import ConnectionTypes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class Histogram:
    name: str
    connections: List[Tuple[float, int]]  # amount of connections and its cnt
    average: float
    median: float


class RandomConnectionsAnalysis:
    """
    this module is in charge of plotting statistics about the random connections
    saves outputs at the same folder with the population data
    assumes ConnectionsTypes have not changed
    """

    __slots__ = ("population_data", "num_of_random_connections",
                 "output_path", "raw_output_folder", "histograms_output_folder")

    def __init__(self, population_data_path):
        # importing population data from path
        self.population_data: PopulationData = PopulationData.import_population_data(population_data_path)
        self.num_of_random_connections: np.ndarray = self.population_data.num_of_random_connections

        # setting output path
        population_data_path: Path = Path(population_data_path) if isinstance(population_data_path, str) else population_data_path
        self.output_path = population_data_path.parent / "random-connections-analysis"
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.raw_output_folder = self.output_path / "raw"
        self.raw_output_folder.mkdir(parents=True, exist_ok=True)

        self.histograms_output_folder = self.output_path / "histograms"
        self.histograms_output_folder.mkdir(parents=True, exist_ok=True)

    # TODO export one with sum
    def export_raw_to_csv(self):
        for con_type in ConnectionTypes:
            data = [{"agent index": index, "connections amount": amount}
                    for index, amount in enumerate(self.num_of_random_connections[:, con_type])]
            df = pd.DataFrame(data=data)
            df.to_csv(self.raw_output_folder / f"raw {con_type.name}.csv", index=False)

    def analyze_connections_histograms(self) -> List[Histogram]:
        histograms = []
        for con_type in ConnectionTypes:
            distribution_of_connections = self.num_of_random_connections[:, con_type]
            average_conn = np.mean(distribution_of_connections)
            median_conn = np.median(distribution_of_connections)
            max_connection = np.max(distribution_of_connections)
            min_connection = np.min(distribution_of_connections)
            connections_histogram, _ = np.histogram(distribution_of_connections,
                                                    bins=math.ceil(max_connection - min_connection + 1),
                                                    range=(min_connection - 0.5, max_connection + 0.5))
            bins = np.arange(min_connection, max_connection + 1).astype(int)
            connections = list(zip(bins, connections_histogram))
            histograms.append(Histogram(name=con_type.name, connections=connections, average=average_conn, median=median_conn))
        return histograms

    def export_histograms_to_csv(self, histograms: List[Histogram]):
        for histogram in histograms:
            df = pd.DataFrame(data=histogram.connections)
            df.to_csv(self.histograms_output_folder / f"{histogram.name} histogram.csv", index=False, header=["connections amount", "cnt"])

    def save_histograms_plots(self, histograms: List[Histogram]):
        for histogram in histograms:
            plt.figure()
            plt.bar([con[0] for con in histogram.connections], [con[1] for con in histogram.connections], width=1)
            plt.ylabel("count")
            plt.xlabel("# of connections")
            plt.xticks()
            plt.title(f"Histogram of connections for connection type {histogram.name}\n \
                      Avg. # = {histogram.average}  ,   Med. # = {histogram.median}")

            plt.savefig(self.histograms_output_folder / f"{histogram.name} histogram.png")
            plt.draw()

    def run_all(self, show: bool = False):
        self.export_raw_to_csv()
        histograms = self.analyze_connections_histograms()
        self.export_histograms_to_csv(histograms)
        self.save_histograms_plots(histograms)
        if show:
            plt.show()


if __name__ == "__main__":
    population_data_path = r"..\inputs\analyze\no-policy-vector-clipped-1\population_data.pickle"
    analyzer = RandomConnectionsAnalysis(population_data_path)
    analyzer.run_all()
