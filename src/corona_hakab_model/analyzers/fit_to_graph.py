from pathlib import Path

import numpy as np
import pandas as pd

from project_structure import OUTPUT_FOLDER


def get_difference_over_time(real_df,
                             df_to_fit,
                             real_column_to_fit="total icu",
                             fitted_column_to_fit="NeedICU So Far"
                             ) -> np.array:
    """
    Gets dataframe of simulation and returns
    the difference between the two datframe at specified columns
    """

    number_of_sim_days = len(df_to_fit.index)
    num_of_origin_days = len(real_df.index)
    pivots_results = np.zeros(shape=(number_of_sim_days - num_of_origin_days + 1, ))
    for i in range(len(pivots_results)):
        current_compare_point = df_to_fit.iloc[i: i + num_of_origin_days]
        compare_now_values = current_compare_point.loc[:, fitted_column_to_fit].values
        origin_now_values = real_df.loc[:, real_column_to_fit].values
        diff = compare_now_values - origin_now_values
        pivots_results[i] = (sum(abs(diff)))
    return pivots_results


def compare_real_to_simulation(real_life_csv, simulation_csv):
    real_df = pd.read_csv(real_life_csv)
    simulation_df = pd.read_csv(simulation_csv)
    differences = get_difference_over_time(real_df, simulation_df)
    pivot = differences.argmin()
    real_life_shiffted = real_df.shift(periods=pivot)
    real_life_result_name = Path(real_life_csv).stem + "_shiftted.csv"
    real_life_result_name = OUTPUT_FOLDER / real_life_result_name
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    real_life_shiffted.to_csv(real_life_result_name)
    print(f"Best fit is at index {pivot}")
    print(f"Shiffted real life csv file is at {real_life_result_name}")
