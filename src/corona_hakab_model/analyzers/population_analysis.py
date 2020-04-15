from os import path
from typing import Tuple
import pickle
import pandas as pd
from generation.connection_types import ConnectionTypes
import matplotlib.pyplot as plt


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

    # test
    print(sick_data_df.Work_guid.dropna().isin(relevant_circles["guid"]).all())
    result_df = pd.merge(left=relevant_circles, right=sick_data_df[relevant_guid_col], how="left", left_on="guid", right_on=relevant_guid_col)
    sick_count_for_relevant_circle = result_df.groupby("guid").count()[relevant_guid_col]
    sick_count_for_relevant_circle.name = "sick_count"

    relevant_circles = relevant_circles.join(sick_count_for_relevant_circle, on="guid")

    # plot histogram
    number_of_works_with_sick = sick_count_for_relevant_circle.where(
        sick_count_for_relevant_circle > 0).count()
    print(f"number of works with at least one sick is {number_of_works_with_sick}"
          f"\ntotal number of works is {len(relevant_circles)}")

    ax1 = plt.subplot(211)
    ax1.plot(relevant_circles.agents_count, relevant_circles.agents_count, label="everyone is sick reference")
    ax1.plot(relevant_circles.agents_count, relevant_circles.sick_count, "ro", label="actual_sick")
    ax1.legend()
    ax1.set_title("number of sick people vs number of workers on workplace")
    plt.xlabel("number of workers on workplace")

    ax2 = plt.subplot(212)
    ax2.plot(relevant_circles.agents_count,
                relevant_circles.sick_count.divide(relevant_circles.agents_count), "bo")
    ax2.set_ylim([0, 1.5])
    ax2.set_title("precetage of sick people vs number of workers on workplace")
    plt.xlabel("number of workers on workplace")
    plt.show()

    work_size_aggregation = relevant_circles.groupby("agents_count").sum()
    plt.plot(work_size_aggregation.index, work_size_aggregation.sick_count, "ro")
    plt.title("total number of sick people for work size")
    plt.ylabel("number of sick people")
    plt.xlabel("number of workers on workplace")
    plt.show()


initial_sick_data_df, final_sick_data_df, all_circles_df = load_population_data_to_dfs()
plot_sick_per_work_size(final_sick_data_df, all_circles_df)

