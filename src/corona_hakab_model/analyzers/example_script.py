from analyzers.agents_sickness_analysis import load_population_data_to_dfs, plot_sick_per_work_size
from corona_hakab_model.analyzers.multi_simulation_analysis import *
from project_structure import SIM_OUTPUT_FOLDER

### This is an example script for running the Simulation Analysis tools

# if you want to analyze all the files in output dir
simulation_output_files = list(Path(SIM_OUTPUT_FOLDER).glob("*.csv"))

# if you want to choose files
# simulation_output_files = [os.path.join(OUTPUT_FOLDER, "20200412-132700.csv")]

# now create a directory with comparison of the parameters returned from the chosen simulations
comparison_dir_path = create_comparison_files(files=simulation_output_files)

# enter parameter name
parameter_name = "AsymptomaticBegin"
csv_path = Path(comparison_dir_path) / (parameter_name + ".csv")

# plot aggregation of the time ,across different runs
plot_minmax_barchart_single_param(csv_path)
# plot different aggregations
plot_aggregations_from_csv(csv_path, aggregation_funcs={"mean": np.mean, "max": np.max})

# plot aggregations of the different runs, across time
# this will create another dir
results_df_dict = create_time_avg_std(comparison_dir_path)
plot_parameter_propagation_aggregated(parameter_names=[parameter_name],
                                      mean_df=results_df_dict["mean"],
                                      std_df=results_df_dict["std"])
# you can also plot all the parameters time propagation
plot_parameter_propagation_aggregated(mean_df=results_df_dict["mean"], std_df=results_df_dict["std"])


# plot social circles sickness analysis
initial_sick_data_df, final_sick_data_df, all_circles_df = load_population_data_to_dfs()
plot_sick_per_work_size(final_sick_data_df, all_circles_df)
plot_sick_per_work_size(initial_sick_data_df, all_circles_df)
