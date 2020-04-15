from os import path
from typing import Tuple
import pickle
import pandas as pd
from generation.connection_types import ConnectionTypes


def load_population_data_to_dfs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_file_path = path.join(path.dirname(path.abspath(__file__)), "../../../output")
    population_data = pickle.load(open(path.join(output_file_path, "population_data.pickle"), "rb"))

    initial_sick_data_df = pd.read_csv(path.join(output_file_path, 'initial_sick.csv'))
    final_sick_data_df = pd.read_csv(path.join(output_file_path, 'all_sick.csv'))

    all_connections = []
    for circle_type, type_connections_list in population_data.social_circles_by_connection_type.items():
        for connection in type_connections_list:
            all_connections.append({"guid": connection.guid,
                                    "agents_count": connection.agent_count,
                                    "connection_type": connection.connection_type})
    all_circles_df = pd.DataFrame(all_connections)
    return initial_sick_data_df, final_sick_data_df, all_circles_df


def plot_sick_per_work_size(sick_data_df: pd.DataFrame, all_circles_df: pd.DataFrame):
    relevant_circles = all_circles_df[all_circles_df["connection_type"] == ConnectionTypes.Work]
    relevant_guid_col = "Work_guid"
    print(sick_data_df.Work_guid.isin(relevant_circles["guid"]).any())
    result_df = pd.merge(left=relevant_circles, right=sick_data_df, how="left", left_on="guid", right_on=relevant_guid_col)
    result_df = result_df.groupby("guid").agg("count")
    # return result_df


initial_sick_data_df, final_sick_data_df, all_circles_df = load_population_data_to_dfs()
print(plot_sick_per_work_size(final_sick_data_df, all_circles_df))

