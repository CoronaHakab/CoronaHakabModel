from os import path
import pickle
import pandas as pd


output_file_path = path.join(path.dirname(path.abspath(__file__)), "../../../output")
population_data = pickle.load(open(path.join(output_file_path, "population_data.pickle"), "rb"))

initial_sick_data_df = pd.read_csv(path.join(output_file_path, 'initial_sick.csv'))
final_sick_data_df = pd.read_csv(path.join(output_file_path, 'all_sick.csv'))

work_circles = population_data.social
circles_data = pd.DataFrame()


