from __future__ import annotations

import logging
import os.path
import random
import sys
from argparse import ArgumentParser
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from corona_hakab_model_data.__data__ import __version__

from application_utils import generate_from_folder, generate_from_master_folder, make_circles_consts, make_matrix_consts
from consts import Consts
from generation.circles_generator import PopulationData
from generation.generation_manager import GenerationManger
from generation.matrix_generator import MatrixData
from manager import SimulationManager
from supervisor import LambdaValueSupervisable, Supervisable

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger('application')
logger.setLevel(logging.INFO)


def main():
    parser = ArgumentParser("COVID-19 Simulation")
    subparser = parser.add_subparsers(dest="sub_command")
    all_parse = subparser.add_parser("all", help="Run both data generation and simulation.")
    gen = subparser.add_parser('generate', help='only generate the population data without running the simulation')
    gen.add_argument("-c",
                     "--circles-consts",
                     dest="circles_consts_path",
                     help="Parameter file with consts for the circles")
    gen.add_argument("-m",
                     "--matrix-consts",
                     dest="matrix_consts_path",
                     help="Parameter file with consts for the matrix")
    gen.add_argument("-o",
                     "--output-folder",
                     dest="output_folder",
                     default='../../output',
                     help="output folder if not using --consts-folder or --master-folder")
    gen.add_argument("--consts-folder",
                     dest="consts_folder",
                     help="Folder to take matrix_consts.json and circles_consts.json from."
                          "Also output folder for generation")
    gen.add_argument("--master-folder",
                     dest="master_folder",
                     help="Master folder - find all immediate sub-folders containing parameter files and generate"
                          "population data and matrix files in them.")
    # Simulation parameters
    sim = subparser.add_parser("simulate", help='Run the simulation using existing data')
    sim.add_argument(
        "-s", "--simulation-parameters", dest="simulation_parameters_path", help="Parameters for simulation engine"
    )
    sim.add_argument('--population-data',
                     dest='population_data',
                     default='../../output/population_data.pickle',
                     help='Previously exported population data file to use in the simulation')
    sim.add_argument('--matrix-data',
                     dest='matrix_data',
                     default='../../output/matrix_data.parasymbolic',
                     help='Previously exported matrix data file to use in the simulation')
    sim.add_argument('--output',
                     dest='output',
                     default='',
                     help='Filepath to resulting csv. Defaults to ../../output/(timestamp).csv')
    sim.add_argument('--figure-path',
                     dest='figure_path',
                     default='',
                     help='Save the resulting figure to a file instead of displaying it')

    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=None,
                        help='Set the random seed. Use only for exact reproducibility. By default, generate new seed.')
    parser.add_argument("--version", action="version", version=__version__)
    args, _ = parser.parse_known_args()
    set_seeds(args.seed)

    if args.sub_command == 'generate':
        generate_data(args)

    if args.sub_command == 'simulate':
        run_simulation(args)

    if args.sub_command == 'all':
        argv_list = sys.argv[1:]
        command_index = argv_list.index('all')
        argv_list[command_index] = 'generate'
        gen_args, _ = parser.parse_known_args(argv_list)
        argv_list[command_index] = 'simulate'
        sim_args, _ = parser.parse_known_args(argv_list)
        generate_data(gen_args)
        run_simulation(sim_args)


def generate_data(args):
    print(args)
    if args.consts_folder:
        generate_from_folder(args.consts_folder)
        return
    elif args.master_folder:
        generate_from_master_folder(args.master_folder)
        return

    circles_consts = make_circles_consts(args.circles_consts_path)

    matrix_consts = make_matrix_consts(args.matrix_consts_path)

    gm = GenerationManger(circles_consts=circles_consts, matrix_consts=matrix_consts)
    gm.save_to_folder(args.output_folder)


def run_simulation(args):
    matrix_data = MatrixData.import_matrix_data(args.matrix_data)
    population_data = PopulationData.import_population_data(args.population_data)
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
                lambda manager: manager.agents_df.n_agents() - manager.medical_machine["Susceptible"].agent_count,
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
    df: pd.DataFrame = sm.dump(filename=args.output)
    df.plot()
    if args.figure_path:
        if not os.path.splitext(args.figure_path)[1]:
            args.figure_path = args.figure_path + '.png'
        plt.savefig(args.figure_path)
    else:
        plt.show()


def set_seeds(seed=0):
    seed = seed or None
    np.random.seed(seed)
    random.seed(seed)


def compare_simulations_example():
    sm1 = SimulationManager(
        (
            Supervisable.Sum(
                "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", "Recovered", "Deceased"
            ),
            "Symptomatic",
            "Recovered",
        ),
        consts=Consts(r0=1.5),
    )
    sm1.run()

    sm2 = SimulationManager(
        (
            Supervisable.Sum(
                "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", "Recovered", "Deceased"
            ),
            "Symptomatic",
            "Recovered",
        ),
        consts=Consts(r0=1.8),
    )
    sm2.run()


if __name__ == "__main__":
    main()
