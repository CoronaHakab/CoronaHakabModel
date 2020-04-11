from __future__ import annotations

from argparse import ArgumentParser
import matplotlib_set_backend
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import os.path

from bsa.universal import write
from consts import Consts
from corona_hakab_model_data.__data__ import __version__
from generation.circles_consts import CirclesConsts
from generation.circles_generator import PopulationData
from generation.generation_manager import GenerationManger
from generation.matrix_consts import MatrixConsts
from generation.matrix_generator import MatrixData
from manager import SimulationManager
from supervisor import LambdaValueSupervisable, Supervisable, SimulationProgression



from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

def main():
    parser = ArgumentParser("COVID-19 Simulation")

    sub_parsers = parser.add_subparsers(dest="sub_command")
    gen = sub_parsers.add_parser("generate", help="only generate the population data without running the simulation")
    gen.add_argument("output")

    parser.add_argument(
        "-s", "--simulation-parameters", dest="simulation_parameters_path", help="Parameters for simulation engine"
    )
    parser.add_argument(
        "-c", "--circles-consts", dest="circles_consts_path", help="Parameter file with consts for the circles"
    )
    parser.add_argument(
        "-m", "--matrix-consts", dest="matrix_consts_path", help="Parameter file with consts for the matrix"
    )
    parser.add_argument('--population-data',
                        dest='population_data',
                        help='Previously exported population data file to use in the simulation')
    parser.add_argument('--matrix-data',
                        dest='matrix_data',
                        help='Previously exported matrix data file to use in the simulation')
    parser.add_argument('--output',
                        dest='output',
                        default='',
                        help='Filepath to resulting csv. Defaults to ../../output/(timestamp).csv')
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=None,
                        help='Set the random seed. Use only for exact reproducibility. By default, generate new seed.')
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()
    set_seeds(args.seed)
    if args.circles_consts_path:
        circles_consts = CirclesConsts.from_file(args.circles_consts_path)
    else:
        circles_consts = CirclesConsts()

    if args.matrix_consts_path:
        matrix_consts = MatrixConsts.from_file(args.matrix_consts_path)
    else:
        matrix_consts = MatrixConsts()

    if args.matrix_data and args.population_data:
        matrix_data = MatrixData.import_matrix_data(args.matrix_data)
        population_data = PopulationData.import_population_data(args.population_data)

    else:
        gm = GenerationManger(circles_consts=circles_consts, matrix_consts=matrix_consts)
        if args.sub_command == 'generate':
            gm.matrix_data.export('matrix')
            return
        population_data = gm.population_data
        matrix_data = gm.matrix_data

    if args.simulation_parameters_path:
        consts = Consts.from_file(args.simulation_parameters_path)
    else:
        consts = Consts()

    set_seeds(args.seed)
    sm = SimulationManager(
        (
            # "Latent",
            Supervisable.State.AddedPerDay("Asymptomatic"),
            Supervisable.State.Current("Asymptomatic"),
            Supervisable.State.TotalSoFar("Asymptomatic"),
            # "Silent",
            # "Asymptomatic",
            # "Symptomatic",
            # "Deceased",
            # "Hospitalized",
            # "ICU",
            # "Susceptible",
            # "Recovered",
            Supervisable.Sum(
                "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", name="currently sick"
            ),
            # LambdaValueSupervisable("ever hospitalized", lambda manager: len(manager.medical_machine["Hospitalized"].ever_visited)),
            LambdaValueSupervisable(
                "was ever sick",
                lambda manager: len(manager.agents) - manager.medical_machine["Susceptible"].agent_count,
            ),
            # Supervisable.NewCasesCounter(),
            # Supervisable.GrowthFactor(
            #    Supervisable.Sum("Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized"),
            Supervisable.NewCasesCounter(),
            LambdaValueSupervisable("Detected Daily", lambda manager: manager.new_detected_daily),
            # LambdaValueSupervisable("Current Confirmed Cases", lambda manager: sum(manager.tested_positive_vector)),
            # Supervisable.R0(),
            # Supervisable.Delayed("Symptomatic", 3),
        ),
        population_data,
        matrix_data,
        consts=consts,
    )
    print(sm)
    sm.run()
    df: pd.DataFrame = sm.dump(args.output)
    df.plot()
    plt.show()


def set_seeds(seed=0):
    seed = seed or None
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()
