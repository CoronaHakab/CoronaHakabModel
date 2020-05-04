from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
import pandas as pd
from pandas import DataFrame


@dataclass
class Simulation:
    path: Path
    prefix: str
    results: List[DataFrame] = field(init=False)

    def __post_init__(self):
        # fills results
        folders = [self.path / folder_name for folder_name in os.listdir(self.path)
                   if
                   (os.path.isdir(self.path / folder_name) and not folder_name == "matrix_analysis")]

        self.results = []
        for folder in folders:
            self.results.append(pd.read_csv(folder / "final_results.csv"))

    def plot_given_stats(self, stats_to_plot: List[str], ax: axis):
        for stat in stats_to_plot:
            means = np.mean([result[stat] for result in self.results], axis=0)
            mins = np.min([result[stat] for result in self.results], axis=0)
            maxs = np.max([result[stat] for result in self.results], axis=0)
            base_line, = ax.plot(means, linewidth=3, label=f"{self.prefix} {stat}")
            ax.fill_between(self.results[0].index, maxs, mins, facecolor=base_line.get_color(), alpha=0.5)


# configuration:
STATS_TO_COMPARE = ["was ever sick"]
TITLE = "multiple matrix only comparisons"
TO_SHOW = True
TO_SAVE = False
SAVE_NAME = TITLE

# simulations folders
father_folder: Path = Path(r"C:\corona\validation\matrix-vs-vector")  # optional
simulations: List[Simulation] = [Simulation(path=father_folder / "no-policy-matrix", prefix="matrix only 1"),
                                 Simulation(path=father_folder / "no-policy-matrix-2", prefix="matrix only 2"),
                                 ]

ax = plt.gca()
for sim in simulations:
    sim.plot_given_stats(STATS_TO_COMPARE, ax)
plt.legend()
plt.title(TITLE)
if TO_SHOW:
    plt.show()
if TO_SAVE:
    plt.savefig(father_folder / SAVE_NAME)
