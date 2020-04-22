from argparse import ArgumentParser
from __data__ import __version__
from project_structure import OUTPUT_FOLDER


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
                     default=OUTPUT_FOLDER,
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
    state_machine = subparser.add_parser("analyze-state-machine", help="Run stochastic analyzer for the state machine")
    state_machine.add_argument("--population_size",
                               dest="population_size",
                               default="50_000",
                               help="Folder to save the result of")
    sim.add_argument('--population-data',
                     dest='population_data',
                     default=OUTPUT_FOLDER / 'population_data.pickle',
                     help='Previously exported population data file to use in the simulation')
    sim.add_argument('--matrix-data',
                     dest='matrix_data',
                     default=OUTPUT_FOLDER / 'matrix_data.parasymbolic',
                     help='Previously exported matrix data file to use in the simulation')
    sim.add_argument('--initial_sick',
                     dest='initial_sick_agents_path',
                     default=OUTPUT_FOLDER / 'initial_sick.csv',
                     help='Output csv file for initial sick agents - after setup of simulation')
    sim.add_argument('--all_sick',
                     dest='all_sick_agents_path',
                     default=OUTPUT_FOLDER / 'all_sick.csv',
                     help='Output csv file for all sick agents - at the end of the simulation run')
    sim.add_argument('--output',
                     dest='output',
                     default='',
                     help='Filepath to resulting csv. Defaults to {}'.format(OUTPUT_FOLDER/'(timestamp).csv'))
    sim.add_argument('--figure-path',
                     dest='figure_path',
                     default='',
                     help='Save the resulting figure to a file instead of displaying it')
    sim.add_argument('--agent-constraints-path',
                     dest='agent_constraints_path',
                     default=None,
                     help='Add constraints to the selection of the initial sick agents, see readme for file format')
    sim.add_argument('--disable_sick_randomization',
                     dest='randomize',
                     action="store_false",
                     default=True,
                     help="Makes the first sick patients the first in the list."
                          " This makes them more connected than random")
    sim.add_argument('--validate-matrix',
                     dest='validate_matrix',
                     action='store_false',
                     help='Validates if the matrix generated is symmetric and all the inputs are probabilities')
    sim.set_defaults(feature=True)
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
