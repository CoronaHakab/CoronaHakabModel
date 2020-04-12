
from __future__ import annotations

import logging
import matplotlib_set_backend
import matplotlib.pyplot as plt
import random
import os.path
import sys
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from bsa.universal import write
from corona_hakab_model_data.__data__ import __version__


import numpy as np

from application_utils import generate_from_folder, generate_from_master_folder, make_circles_consts, make_matrix_consts
from consts import Consts
from generation.circles_generator import PopulationData
from generation.generation_manager import GenerationManger
from generation.matrix_generator import MatrixData
from manager import SimulationManager
from subconsts.modules_argpasers import get_simulation_args_parser
from supervisor import LambdaValueSupervisable, Supervisable


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

from matrix_analysis.MatrixAnalysis import MatrixAnalyzer


def main():
    parser = ArgumentParser("COVID-19 Simulation")

    sub_parsers = parser.add_subparsers(dest='sub_command')
    gen = sub_parsers.add_parser('generate', help='only generate the population data without running the simulation')
    gen.add_argument('output')

    matrix = sub_parsers.add_parser('analyze-matrix', help="analyze matrix histograms and export csv's")
    matrix.add_argument("--matrix",
                        dest="matrix_path",
                        help="Matrix file to analyze")
    matrix.add_argument("--show",
                        dest="show",
                        action="store_true",
                        help="Show histograms")

    parser.add_argument("-s",
                        "--simulation-parameters",
                        dest="simulation_parameters_path",
                        help="Parameters for simulation engine")
    parser.add_argument("-c",
                        "--circles-consts",
                        dest="circles_consts_path",
                        help="Parameter file with consts for the circles")
    parser.add_argument("-m",
                        "--matrix-consts",
                        dest="matrix_consts_path",
                        help="Parameter file with consts for the matrix")
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args()

    if args.sub_command == 'analyze-matrix':
        matrix_analyzer = MatrixAnalyzer(args.matrix_path)
        matrix_analyzer.export_raw_matrices_to_csv()
        matrix_analyzer.analyze_histograms()
        if args.show:
            plt.show()
        return

    if args.circles_consts_path:
        circles_consts = CirclesConsts.from_file(args.circles_consts_path)
    else:
        circles_consts = CirclesConsts()


def main():
    logger = logging.getLogger('application')
    logger.setLevel(logging.INFO)
    parser = get_simulation_args_parser()
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
                lambda manager: len(manager.agents) - manager.medical_machine["Susceptible"].agent_count,
            ),
            Supervisable.NewCasesCounter(),
            Supervisable.Wrappers.Growth(Supervisable.NewCasesCounter(), 1),
            Supervisable.Wrappers.RunningAverage(Supervisable.Wrappers.Growth(Supervisable.NewCasesCounter()), 7),
            Supervisable.Wrappers.Growth(Supervisable.NewCasesCounter(), 7),
            # Supervisable.GrowthFactor(
            #    Supervisable.Sum("Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized"),
            # LambdaValueSupervisable("Detected Daily", lambda manager: manager.new_detected_daily),
            # LambdaValueSupervisable("Current Confirmed Cases", lambda manager: sum(manager.tested_positive_vector)),
            # Supervisable.R0(),
            # Supervisable.Delayed("Symptomatic", 3),
        ),
        population_data,
        matrix_data,
        run_args=args,
        consts=consts,
    )
    print(sm)
    sm.run()
    df: pd.DataFrame = sm.dump(filename=args.output)
    df.plot()
    if args.figure_path:
        if not os.path.splitext(args.figure_path)[1]:
            args.figure_path = args.figure_path+'.png'
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
