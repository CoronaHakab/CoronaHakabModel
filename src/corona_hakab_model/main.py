from __future__ import annotations

import logging
from argparse import ArgumentParser
import matplotlib_set_backend
import matplotlib.pyplot as plt

from bsa.universal import write
from application_utils import generate_from_folder, generate_from_master_folder
from consts import Consts
from corona_hakab_model_data.__data__ import __version__
from generation.circles_consts import CirclesConsts
from generation.generation_manager import GenerationManger
from generation.matrix_consts import MatrixConsts
from manager import SimulationManager
from supervisor import LambdaValueSupervisable, Supervisable, SimulationProgression


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger('application')
logger.setLevel(logging.INFO)


def main():
    parser = ArgumentParser("COVID-19 Simulation")

    # Generation parameters:
    subparser = parser.add_subparsers(dest="sub_command")
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
    parser.add_argument("-s",
                        "--simulation-parameters",
                        dest="simulation_parameters_path",
                        help="Parameters for simulation engine.")

    parser.add_argument("--matrix",
                        dest="matrix_file_path",
                        help="pre-generated matrix file path.")
    parser.add_argument("--population-data",
                        dest="population_data_file_path",
                        help="pre-generated matrix file path.")
    args = parser.parse_args()

    if args.sub_command == 'generate':
        generate_command(args)
        return

    if args.circles_consts_path:
        circles_consts = CirclesConsts.from_file(args.circles_consts_path)
    else:
        circles_consts = CirclesConsts()

    if args.matrix_consts_path:
        matrix_consts = MatrixConsts.from_file(args.matrix_consts_path)
    else:
        matrix_consts = MatrixConsts()

    gm = GenerationManger(circles_consts=circles_consts, matrix_consts=matrix_consts)

    if args.sub_command == "generate":
        with open(args.output, "wb") as w:
            write(gm.matrix_data.matrix, w)
        return

    if args.simulation_parameters_path:
        consts = Consts.from_file(args.simulation_parameters_path)
    else:
        consts = Consts()

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
        gm.population_data,
        gm.matrix_data,
        consts=consts,
    )
    print(sm)
    sm.run()
    df: pd.DataFrame = sm.dump()
    df.plot()
    plt.show()


def generate_command(args):
    if args.consts_folder:
        generate_from_folder(args.consts_folder)
        return
    elif args.master_folder:
        generate_from_master_folder(args.master_folder)
        return

    if not args.output_folder:
        logger.error("No output folder given! use --output-folder")

    if args.circles_consts_path:
        circles_consts = CirclesConsts.from_file(args.circles_consts_path)
    else:
        circles_consts = CirclesConsts()

    if args.matrix_consts_path:
        matrix_consts = MatrixConsts.from_file(args.matrix_consts_path)
    else:
        matrix_consts = MatrixConsts()

    gm = GenerationManger(circles_consts=circles_consts, matrix_consts=matrix_consts)
    gm.save_to_folder(args.output_folder)


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
