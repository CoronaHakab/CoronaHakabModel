from __future__ import annotations

import logging
import random
import os.path
import sys
from matplotlib import pyplot as plt

import numpy as np

from analyzers.state_machine_analysis import extract_state_machine_analysis
from application_utils import generate_from_folder, generate_from_master_folder, make_circles_consts, make_matrix_consts
from consts import Consts
from generation.circles_generator import PopulationData
from generation.generation_manager import GenerationManger
from generation.matrix_generator import MatrixData
from generation.connection_types import ConnectionTypes
from manager import SimulationManager
from agent import InitialAgentsConstraints
from subconsts.modules_argpasers import get_simulation_args_parser
from supervisor import LambdaValueSupervisable, Supervisable
from analyzers import matrix_analysis


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd


def main():
    logger = logging.getLogger('application')
    logger.setLevel(logging.INFO)
    parser = get_simulation_args_parser()

    args, _ = parser.parse_known_args()
    set_seeds(args.seed)

    if args.sub_command == 'analyze-state-machine':
        extract_state_machine_analysis(vars(args))

    if args.sub_command == 'analyze-matrix':
        analyze_matrix(args)

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
    initial_agent_constraints = InitialAgentsConstraints(args.agent_constraints_path)
    if args.simulation_parameters_path:
        consts = Consts.from_file(args.simulation_parameters_path)
    else:
        consts = Consts()
    set_seeds(args.seed)
    sm = SimulationManager(
        (
            # "Latent",
            Supervisable.State.AddedPerDay("AsymptomaticBegin"),
            Supervisable.State.Current("AsymptomaticBegin"),
            Supervisable.State.TotalSoFar("AsymptomaticBegin"),
            # "Silent",
            # "Asymptomatic",
            # "Symptomatic",
            # "Deceased",
            # "Hospitalized",
            # "ICU",
            # "Susceptible",
            # "Recovered",
            Supervisable.Sum(
                "Latent",
                "Latent-Asymp",
                "Latent-Presymp",
                "AsymptomaticBegin",
                "AsymptomaticEnd",
                "Pre-Symptomatic",
                "Mild-Condition",
                "NeedOfCloseMedicalCare",
                "NeedICU",
                "ImprovingHealth",
                "PreRecovered",
                name="currently sick"
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
            Supervisable.CurrentInfectedTable(interval=consts.export_infected_agents_interval),
            # LambdaValueSupervisable("Detected Daily", lambda manager: manager.new_detected_daily),
            # LambdaValueSupervisable("Current Confirmed Cases", lambda manager: sum(manager.tested_positive_vector)),
            # Supervisable.R0(),
            # Supervisable.Delayed("Symptomatic", 3),
            LambdaValueSupervisable("daily infected by work", lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Work]),
            LambdaValueSupervisable("daily infected by school", lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.School]),
            LambdaValueSupervisable("daily infected by other", lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Other]),
            LambdaValueSupervisable("daily infected by family", lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Family]),
            LambdaValueSupervisable("daily infected by kindergarten", lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Kindergarten]),
            LambdaValueSupervisable("daily infected by synagogue", lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Synagogue]),
            LambdaValueSupervisable("daily infections from Latent infector", lambda manager: manager.new_sick_by_infector_medical_state["Latent"]),
            LambdaValueSupervisable("daily infections from Latent-Asymp infector", lambda manager: manager.new_sick_by_infector_medical_state["PreRecovered"]),
            LambdaValueSupervisable("daily infections from Latent-Presymp infector", lambda manager: manager.new_sick_by_infector_medical_state["Latent-Asymp"]),
            LambdaValueSupervisable("daily infections from AsymptomaticBegin infector", lambda manager: manager.new_sick_by_infector_medical_state["AsymptomaticBegin"]),
            LambdaValueSupervisable("daily infections from AsymptomaticEnd infector", lambda manager: manager.new_sick_by_infector_medical_state["AsymptomaticEnd"]),
            LambdaValueSupervisable("daily infections from Pre-Symptomatic infector", lambda manager: manager.new_sick_by_infector_medical_state["Pre-Symptomatic"]),
            LambdaValueSupervisable("daily infections from Mild-Condition infector", lambda manager: manager.new_sick_by_infector_medical_state["Mild-Condition"]),
            LambdaValueSupervisable("daily infections from NeedOfCloseMedicalCare infector", lambda manager: manager.new_sick_by_infector_medical_state["NeedOfCloseMedicalCare"]),
            LambdaValueSupervisable("daily infections from NeedICU infector", lambda manager: manager.new_sick_by_infector_medical_state["NeedICU"]),
            LambdaValueSupervisable("daily infections from ImprovingHealth infector", lambda manager: manager.new_sick_by_infector_medical_state["ImprovingHealth"]),
            LambdaValueSupervisable("daily infections from PreRecovered infector", lambda manager: manager.new_sick_by_infector_medical_state["PreRecovered"]),
        ),
        population_data,
        matrix_data,
        initial_agent_constraints,
        run_args=args,
        consts=consts,
    )
    print(sm)
    sm.run()
    df: pd.DataFrame = sm.dump(filename=args.output)
    df.iloc[:, :-16].plot()
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


def analyze_matrix(args):
    matrix_data = matrix_analysis.import_matrix_data(args.matrix_path)
    matrix_analysis.export_raw_matrices_to_csv(matrix_data)
    histograms = matrix_analysis.analyze_histograms(matrix_data)
    matrix_analysis.export_histograms(histograms)
    matrix_analysis.save_histogram_plots(histograms)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()