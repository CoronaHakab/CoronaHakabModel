import json
from datetime import datetime
from agent import InitialAgentsConstraints
from analyzers.state_machine_analysis import monte_carlo_state_machine_analysis
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
    df = pd.DataFrame(columns=range(1, max_number_of_days+1),
                      index=range(max_loops))
    total_loops = 0
    for number_of_days, number_of_loops in sorted(simulation_time_slices.items(), reverse=True):
        for i in range(total_loops, number_of_loops):
            sm.setup_sick()
            current_run_results = _compute_simulation_new_sick(sm, number_of_days)
            df.iloc[i] = pd.Series(current_run_results)
            sm.reset()
        total_loops += number_of_loops
    return df


def _calculate_series_r0(infections, p_tau):
    weighted_infections_history = sum(infections[:-1] * p_tau)
    if weighted_infections_history == 0:
        return 0
    return infections.iloc[-1] / weighted_infections_history


def r_effective_over_time(infections_df, config) -> pd.DataFrame:
    p_tau_reversed = np.array(config['p_tau'][::-1])  # The distribution to get infected on each day from agent
    interval_length = len(p_tau_reversed)
    r_effective_days = list(range(interval_length, len(infections_df.columns)))
    r_effective_df = pd.DataFrame(np.nan,
                                  index=infections_df.index,
                                  columns=r_effective_days)
    for i in r_effective_days:
        average_total_infections = infections_df.loc[:, i-interval_length: i-1] @ p_tau_reversed
        r_effective_df.loc[:, i] = infections_df.loc[:, i]/average_total_infections

    return r_effective_df


def calculate_r_effective(config):
    infect_statistics = new_infects_over_time(config)
    file_suffix = datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    infect_statistics.to_csv(OUTPUT_FOLDER /
                             ("multiple_run_infection_statistics_"+file_suffix))
    r0_over_time = r_effective_over_time(infect_statistics, config)
    r0_over_time.to_csv(OUTPUT_FOLDER /
                        ("multiple_run_r0_over_time_"+file_suffix))
    r0_per_day_averages = r0_over_time.mean(axis=1)
    r0_per_day_averages.to_csv(OUTPUT_FOLDER /
                               ("r0_averaged"+file_suffix))


if __name__ == "__main__":
    calculate_r_effective()
