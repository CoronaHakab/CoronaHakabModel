from argparse import ArgumentParser
from corona_hakab_model_data.__data__ import __version__


def get_simulation_args_parser():
    """
    Returns a args parser used by the simulation
    """
    parser = ArgumentParser("COVID-19 Simulation")
    subparser = parser.add_subparsers(dest="sub_command")
    subparser.add_parser("all", help="Run both data generation and simulation.")
    matrix = subparser.add_parser('analyze-matrix', help="analyze matrix histograms and export csv's")
    matrix.add_argument("--matrix",
                        dest="matrix_path",
                        help="Matrix file to analyze")
    matrix.add_argument("--show",
                        dest="show",
                        action="store_true",
                        help="Show histograms")
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
    sim.add_argument('--initial_sick',
                     dest='initial_sick_agents_path',
                     default='../../output/initial_sick.csv',
                     help='Output csv file for initial sick agents - after setup of simulation')
    sim.add_argument('--all_sick',
                     dest='all_sick_agents_path',
                     default='../../output/all_sick.csv',
                     help='Output csv file for all sick agents - at the end of the simulation run')
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
    return parser


def get_default_simulation_args_values():
    """
    Returns the default Namespace object that simulation uses
    """
    arg_parser = get_simulation_args_parser()
    return arg_parser.parse_args(["all"])
