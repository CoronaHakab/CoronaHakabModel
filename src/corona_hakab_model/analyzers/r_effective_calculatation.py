import json
from datetime import datetime
from analyzers.state_machine_analysis import monte_carlo_state_machine_analysis
from common.agent import InitialAgentsConstraints
from project_structure import OUTPUT_FOLDER, ANALYZERS_FOLDER
from consts import Consts
from generation.circles_consts import CirclesConsts
from generation.circles_generator import PopulationData, CirclesGenerator
from generation.matrix_consts import MatrixConsts
from generation.matrix_generator import MatrixData, MatrixGenerator
from manager import SimulationManager
import pandas as pd
from subconsts.modules_argpasers import get_default_silent_simulation_args
import numpy as np


def _compute_simulation_new_sick(sm, number_of_step):
    """

    :param sm: A SimulationManager object
    :param number_of_step: Number of steps to run the manager
    :return:
    """
    new_sick_every_day = np.zeros((number_of_step,))
    for i in range(number_of_step):
        sm.step()
        new_sick_every_day[i] = sm.new_sick_counter
    return new_sick_every_day


def new_infects_over_time(config=None) -> pd.DataFrame:
    if config is None:
        config = dict()

    simulation_time_slices = config['simulation_time_slices']

    if 'population_data_path' in config:
        population_data = PopulationData.import_population_data(config['population_data_path'])
    else:
        circle_consts = CirclesConsts(population_size=config['circle_consts_config']['population_size'],)
        population_data = CirclesGenerator(circle_consts).population_data

    if 'matrix_data_path' in config:
        matrix_data = MatrixData.import_matrix_data(config['matrix_data_path'])
    else:
        matrix_consts = MatrixConsts()
        matrix_data = MatrixGenerator(population_data=population_data,
                                      matrix_consts=matrix_consts).matrix_data
    if 'consts_file' in config:
        consts = Consts.from_file(config['consts_file'])
    else:
        if 'consts_config' in config:
            consts = Consts(**config['consts_config'])
        else:
            consts = Consts()
        if 'initial_agent_constraints' in config:
            initial_agent_constraint = InitialAgentsConstraints(config['initial_agent_constraints'])
        else:
            initial_agent_constraint = InitialAgentsConstraints()

    sim_run_args = get_default_silent_simulation_args()

    sm = SimulationManager((),
                           population_data,
                           matrix_data,
                           inital_agent_constraints=initial_agent_constraint,
                           run_args=sim_run_args,
                           consts=consts)
    max_number_of_days = max(simulation_time_slices.keys())
    max_loops = max(simulation_time_slices.values())
    df = pd.DataFrame(columns=range(1, max_number_of_days),
                      index=range(max_loops))
    total_loops = 0
    for number_of_days, number_of_loops in sorted(simulation_time_slices.items(), reverse=True):
        for i in range(total_loops, number_of_loops):
            sm.setup_sick()
            current_run_results = _compute_simulation_new_sick(sm, number_of_days)
            df.iloc[i] = pd.Series(current_run_results)
            sm.reset()
        total_loops = number_of_loops
    return df


def weighted_r_effective_over_time(infections_df, config) -> pd.DataFrame:
    p_tau_reversed = np.array(config['p_tau'][::-1])  # The distribution to get infected on each day from agent
    interval_length = len(p_tau_reversed)
    first_column, last_column = infections_df.columns[0], infections_df.columns[-1]
    r_effective_days = list(range(interval_length+first_column, last_column+1))
    r_effective_df = pd.DataFrame(np.nan,
                                  index=infections_df.index,
                                  columns=r_effective_days)
    for i in r_effective_days:
        average_total_infections = infections_df.loc[:, i-interval_length: i-1] @ p_tau_reversed
        r_effective_df.loc[:, i] = infections_df.loc[:, i]/average_total_infections

    return r_effective_df


def simplified_r_effective_over_time(infections_df, config):
    state_machine_stats = monte_carlo_state_machine_analysis(config['monte_carlo_config'])
    total_expected_infected_days = state_machine_stats["average_time_to_terminal"]
    r_effective_days = infections_df.columns[1:]
    r_effective_df = pd.DataFrame(np.nan,
                                  index=infections_df.index,
                                  columns=r_effective_days)
    for i in r_effective_days:
        yesterday_infects = infections_df.loc[:, i-1].replace(0, np.nan)
        r0_of_day_series = infections_df.loc[:, i]/yesterday_infects
        r_effective_df.loc[:, i] = r0_of_day_series ** total_expected_infected_days
    return r_effective_df


def _sim_r_effective_calc_by_type(infect_statistics, config, file_suffix):
    if config["r_effective_computation_type"] == "all":
        config["r_effective_computation_type"] = "simple"
        _sim_r_effective_calc_by_type(infect_statistics, config, file_suffix)
        config["r_effective_computation_type"] = "weighted"
        _sim_r_effective_calc_by_type(infect_statistics, config, file_suffix)
    else:
        if config["r_effective_computation_type"] == "simple":
            r_effective_over_time = simplified_r_effective_over_time(infect_statistics,
                                                                     config)
        elif config["r_effective_computation_type"] == "weighted":
            r_effective_over_time = weighted_r_effective_over_time(infect_statistics,
                                                                   config)
        else:
            raise NotImplementedError
        file_prefix = config["r_effective_computation_type"]
        OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
        OUTPUT_FOLDER.r_effective_over_time.to_csv(OUTPUT_FOLDER /
                                     (f"{file_prefix}_"
                                      f"multiple_run_r_eff_over_time_"
                                      f"{file_suffix}"))
        r0_per_day_averages = r_effective_over_time.mean()
        r0_per_day_averages.to_csv(OUTPUT_FOLDER /
                                   (f"{file_prefix}_"
                                    f"r_eff_averaged_"
                                    f"{file_suffix}"))


def calculate_r_effective(*, config=None, json_file=None):
    assert json_file or config, "Must give a path to json file or a config dict as input"
    if json_file:
        with open(json_file, "r") as fh:
            config = json.load(fh)
        # We make a list o/w the iterator updates
        for k in list(config['simulation_time_slices'].keys()):
            config['simulation_time_slices'][int(k)] = config['simulation_time_slices'].pop(k)
    infect_statistics = new_infects_over_time(config)
    file_suffix = datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    infect_statistics.to_csv(OUTPUT_FOLDER /
                             ("multiple_run_infection_statistics_"+file_suffix))
    _sim_r_effective_calc_by_type(infect_statistics,
                                  config,
                                  file_suffix)


if __name__ == "__main__":
    main_config_file = ANALYZERS_FOLDER / "r_effective_calculation_default_config.json"
    calculate_r_effective(json_file=main_config_file)