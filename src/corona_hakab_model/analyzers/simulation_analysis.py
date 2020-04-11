import os
from glob import glob
from project_structure import OUTPUT_FOLDER
import pandas as pd
from datetime import datetime


def create_comparison_files(parameters_to_compare: list, files: list = None):
    """

    :param parameters_to_compare: List of strings, each string is a name of column
    :param files: Files to compare. If not given, takes all the files in the output folder.
    :return: N/A. For each parameter creates a file combining the information from each file.
    """
    if files is None:
        files = glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
    dfs_dict = dict(map(lambda f: (os.path.basename(f), pd.read_csv(f)), files))
    result_folder_name = os.path.join(OUTPUT_FOLDER, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    os.mkdir(result_folder_name)
    for parameter in parameters_to_compare:
        result_df = pd.DataFrame(columns=dfs_dict.keys())
        for file_name, file_df in dfs_dict.items():
            result_df.loc[:, file_name] = file_df.loc[:, parameter]
        result_df.to_csv(os.path.join(result_folder_name, f"{parameter}.csv"))
