from __future__ import annotations

import logging
import random
import os.path
from pathlib import Path
import sys
from matplotlib import pyplot as plt

import numpy as np

from analyzers.fit_to_graph import compare_real_to_simulation
from analyzers.state_machine_analysis import extract_state_machine_analysis
from common.application_utils import generate_from_folder, generate_from_master_folder, make_circles_consts, \
    make_matrix_consts
from common.isolation_types import IsolationTypes
from consts import Consts
from generation.circles_generator import PopulationData
from generation.generation_manager import GenerationManger
from generation.matrix_generator import MatrixData, ConnectionData
from generation.connection_types import ConnectionTypes
from manager import SimulationManager
from common.agent import InitialAgentsConstraints
from subconsts.modules_argpasers import get_simulation_args_parser
from supervisor import LambdaValueSupervisable, Supervisable
from analyzers import matrix_analysis
from analyzers.random_connections_analysis import RandomConnectionsAnalysis

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

    if args.sub_command == 'analyze-random-connections':
        analyze_random_connections(args)

    if args.sub_command == 'shift-real-life':
        sys.argv = sys.argv[1:]
        assert len(sys.argv) == 3, f"Gave {len(sys.argv)} parameters. Needs to give 2 parameters as input"
        compare_real_to_simulation(sys.argv[1],
                                   sys.argv[2])

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

    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

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
    print(args)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    matrix_data = MatrixData.import_matrix_data(args.matrix_data)
    population_data = PopulationData.import_population_data(args.population_data)
    connection_data = ConnectionData.import_connection_data(args.connection_data)
    initial_agent_constraints = InitialAgentsConstraints(args.agent_constraints_path)
    if args.simulation_parameters_path:
        consts = Consts.from_file(args.simulation_parameters_path)
    else:
        consts = Consts()
    set_seeds(args.seed)
    sm = SimulationManager(
        (
            # "Latent",
            Supervisable.State.TotalSoFar("AsymptomaticBegin"),
            Supervisable.State.TotalSoFar("Deceased"),
            Supervisable.State.TotalSoFar("NeedOfCloseMedicalCare"),
            Supervisable.State.TotalSoFar("NeedICU"),
            Supervisable.State.TotalSoFar("Mild-Condition-Begin"),
            Supervisable.State.TotalSoFar("Mild-Condition-End"),

            Supervisable.State.AddedPerDay("AsymptomaticBegin"),
            Supervisable.State.AddedPerDay("Deceased"),
            Supervisable.State.AddedPerDay("NeedOfCloseMedicalCare"),
            Supervisable.State.AddedPerDay("Latent-Asymp"),
            Supervisable.State.AddedPerDay("Latent-Presymp"),
            Supervisable.State.AddedPerDay("NeedICU"),
            Supervisable.State.AddedPerDay("Recovered"),
            Supervisable.State.AddedPerDay("Mild-Condition-Begin"),
            Supervisable.State.AddedPerDay("Mild-Condition-End"),

            Supervisable.State.Current("NeedOfCloseMedicalCare"),
            Supervisable.State.Current("AsymptomaticBegin"),
            Supervisable.State.Current("Latent-Asymp"),
            Supervisable.State.Current("Latent-Presymp"),
            Supervisable.State.Current("Pre-Symptomatic"),
            Supervisable.State.Current("NeedICU"),
            Supervisable.State.Current("Recovered"),
            Supervisable.State.Current("Mild-Condition-Begin"),
            Supervisable.State.Current("Mild-Condition-End"),
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
                "Mild-Condition-Begin",
                "Mild-Condition-End",
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
            # Supervisable.AppliedPolicyReportSupervisable(),
            # LambdaValueSupervisable("Detected Daily", lambda manager: manager.new_detected_daily),
            # LambdaValueSupervisable("Current Confirmed Cases", lambda manager: sum(manager.tested_positive_vector)),
            # Supervisable.R0(),
            # Supervisable.Delayed("Symptomatic", 3),
            LambdaValueSupervisable("daily infected by work",
                                    lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Work]),
            LambdaValueSupervisable("daily infected by school",
                                    lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.School]),
            LambdaValueSupervisable("daily infected by other",
                                    lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Other]),
            LambdaValueSupervisable("daily infected by family",
                                    lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Family]),
            LambdaValueSupervisable("daily infected by kindergarten",
                                    lambda manager: manager.new_sick_by_infection_method[ConnectionTypes.Kindergarten]),
            LambdaValueSupervisable("daily infections from Latent infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["Latent"]),
            LambdaValueSupervisable("daily infections from Latent-Asymp infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["PreRecovered"]),
            LambdaValueSupervisable("daily infections from Latent-Presymp infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["Latent-Asymp"]),
            LambdaValueSupervisable("daily infections from AsymptomaticBegin infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["AsymptomaticBegin"]),
            LambdaValueSupervisable("daily infections from AsymptomaticEnd infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["AsymptomaticEnd"]),
            LambdaValueSupervisable("daily infections from Pre-Symptomatic infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["Pre-Symptomatic"]),
            LambdaValueSupervisable("daily infections from Mild-Condition-Begin infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["Mild-Condition-Begin"]),
            LambdaValueSupervisable("daily infections from Mild-Condition-End infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["Mild-Condition-End"]),
            LambdaValueSupervisable("daily infections from NeedOfCloseMedicalCare infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state[
                                        "NeedOfCloseMedicalCare"]),
            LambdaValueSupervisable("daily infections from NeedICU infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["NeedICU"]),
            LambdaValueSupervisable("daily infections from ImprovingHealth infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["ImprovingHealth"]),
            LambdaValueSupervisable("daily infections from PreRecovered infector",
                                    lambda manager: manager.new_sick_by_infector_medical_state["PreRecovered"]),
            LambdaValueSupervisable("Isolated",
                                    lambda manager: np.count_nonzero(
                                        manager.agents_in_isolation != IsolationTypes.NONE)),
            LambdaValueSupervisable("Isolated Hotel",
                                    lambda manager: np.count_nonzero(
                                        manager.agents_in_isolation == IsolationTypes.HOTEL)),
            LambdaValueSupervisable("Isolated Home",
                                    lambda manager: np.count_nonzero(
                                        manager.agents_in_isolation == IsolationTypes.HOME)),
            LambdaValueSupervisable("Got out of isolation - due date",
                                    lambda manager: manager.left_isolation_by_reason['due_date']),
            LambdaValueSupervisable("Got out of isolation - many negative tests",
                                    lambda manager: manager.left_isolation_by_reason['negative_tests']),
            LambdaValueSupervisable("Number of tests",
                                    lambda manager: manager.healthcare_manager.num_of_tested),
        ),
        population_data,
        matrix_data,
        connection_data,
        initial_agent_constraints,
        run_args=args,
        consts=consts,
    )
    print(sm)
    sm.run()
    df: pd.DataFrame = sm.dump(filename=args.output)
    # using parent since args.output gives the sim_records folder
    consts.export(export_path=Path(args.output).parent, file_name="simulation_consts.json")
    df.plot()
    if args.figure_path:
        if not os.path.splitext(args.figure_path)[1]:
            args.figure_path = args.figure_path + '.png'
        plt.savefig(args.figure_path)

    if args.show_plot:
        plt.show()


def set_seeds(seed=0):
    seed = seed or None
    np.random.seed(seed)
    random.seed(seed)


def analyze_matrix(args):
    matrices = matrix_analysis.import_matrix_as_csr(args.matrix_path)
    matrix_analysis.export_raw_matrices_to_csv(matrices)
    histograms = matrix_analysis.analyze_histograms(matrices)
    matrix_analysis.export_histograms(histograms)
    matrix_analysis.save_histogram_plots(histograms)
    if args.show:
        plt.show()


def analyze_random_connections(args):
    analyzer = RandomConnectionsAnalysis(args.population_data_path)
    analyzer.run_all(args.show)


if __name__ == "__main__":
    main()
