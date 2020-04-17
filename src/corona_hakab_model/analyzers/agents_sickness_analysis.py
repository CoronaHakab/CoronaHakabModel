from pathlib import Path
from typing import Tuple
import pandas as pd
from generation.connection_types import ConnectionTypes
import matplotlib.pyplot as plt
from analyzers.config import POPULATION_OUTPUT_PATH
from generation.circles_generator import PopulationData


def load_population_data_to_dfs(output_file_path=POPULATION_OUTPUT_PATH) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    population_data = PopulationData.import_population_data(Path(output_file_path) / 'population_data.pickle')
    initial_sick_data_df = pd.read_csv(Path(output_file_path) / 'initial_sick.csv')
    final_sick_data_df = pd.read_csv(Path(output_file_path) / 'all_sick.csv')

    all_connections = []
    for circle_type, type_connections_list in population_data.social_circles_by_connection_type.items():
        for connection in type_connections_list:
            all_connections.append({"guid": connection.guid,
                                    "agents_count": connection.agent_count,
                                    "connection_type": connection.connection_type})
    all_circles_df = pd.DataFrame(all_connections)
    return initial_sick_data_df, final_sick_data_df, all_circles_df


def plot_sick_per_work_size(sick_data_df: pd.DataFrame, all_circles_df: pd.DataFrame, log_scale=False):
    relevant_circles = all_circles_df[all_circles_df["connection_type"] == ConnectionTypes.Work]
    relevant_circles = relevant_circles.sort_values(by=["agents_count"])
    relevant_guid_col = "Work_guid"
    relevant_circles_name = "work"

    # test
    # print(sick_data_df[relevant_guid_col].dropna().isin(relevant_circles["guid"]).all())
    relevant_circles = add_sick_count_col_to_relevant_circles(relevant_circles, sick_data_df, relevant_guid_col)
    plot_sick_per_relevant_circle(relevant_circles, relevant_circles_name, log_scale)


def add_sick_count_col_to_relevant_circles(relevant_circles: pd.DataFrame, sick_agents: pd.DataFrame,
                                          relevant_guid_col: str) -> pd.DataFrame:
    result_df = pd.merge(left=relevant_circles, right=sick_agents[relevant_guid_col],
                         how="left", left_on="guid", right_on=relevant_guid_col)
    sick_count_for_relevant_circle = result_df.groupby("guid").count()[relevant_guid_col]
    sick_count_for_relevant_circle.name = "sick_count"
    return relevant_circles.join(sick_count_for_relevant_circle, on="guid")


def plot_sick_per_relevant_circle(relevant_circles: pd.DataFrame, relevant_circles_name: str, log_scale=False):
    # total number of s
    number_of_works_with_sick = relevant_circles["sick_count"].where(
        relevant_circles["sick_count"] > 0).count()
    print(f"number of {relevant_circles_name}s with at least one sick is {number_of_works_with_sick}"
          f"\ntotal number of {relevant_circles_name}s is {len(relevant_circles)}")

    # plot
    ax1 = plt.subplot(211)
    ax1.plot(relevant_circles.agents_count, relevant_circles.agents_count, label="everyone is sick reference")
    ax1.plot(relevant_circles.agents_count, relevant_circles.sick_count, "ro", label="actual_sick")
    ax1.legend()
    ax1.set_title(f"number of sick people vs number of agents in {relevant_circles_name}")
    plt.xlabel(f"number of agents in {relevant_circles_name}")
    plt.ylabel("number of sick people")
    if log_scale:
        plt.xscale("log")

    ax2 = plt.subplot(212)
    ax2.plot(relevant_circles.agents_count,
                relevant_circles.sick_count.divide(relevant_circles.agents_count), "bo")
    ax2.set_ylim([0, 1.5])
    ax2.set_title(f"percentage of sick people vs number of agents in {relevant_circles_name}")
    plt.xlabel(f"number of agents in {relevant_circles_name}")
    plt.ylabel("percentage of sick people")
    if log_scale:
        plt.xscale("log")
    plt.show()

    work_size_aggregation = relevant_circles.groupby("agents_count").sum()
    plt.plot(work_size_aggregation.index, work_size_aggregation.sick_count, "ro")
    plt.title(f"total number of sick people for {relevant_circles_name} size")
    plt.xlabel(f"number of agents in {relevant_circles_name}")
    plt.ylabel("number of sick people")
    if log_scale:
        plt.xscale("log")
    plt.show()

