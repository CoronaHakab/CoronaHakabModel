from consts import Consts
from generation.circles_consts import CirclesConsts
from generation.circles_generator import PopulationData, CirclesGenerator
from generation.matrix_consts import MatrixConsts
from generation.matrix_generator import MatrixData, MatrixGenerator

# This dict's keys is number of days and value is number of simulation run
# For example for key of 10 and value of 50 we run the first 10 days of the simulation 50 times
_DEFAULT_SIMULATION_TIME_SLICES = {
    10: 50,
    20: 30,
    40: 10
}


def _compute_r0_for_one_run(sm, number_of_step):
    new_sick_every_day = list()
    growth_rates = list()
    for i in range(number_of_step):
        sm.step()
        new_sick_every_day.append(sm.new_sick_counter)
        if len(new_sick_every_day) > 1:
            if new_sick_every_day[i - 1] > 0:
                growth_rates.append(new_sick_every_day[i] / new_sick_every_day[i - 1])
            else:
                growth_rates.append(0)
    return dict(new_sick_every_day=new_sick_every_day, growth_rates=growth_rates)


def calculate_r_effective(config=dict()):
    if 'simulation_time_slices' in config:
        simulation_time_slices = config['simulation_time_slices']
    else:
        simulation_time_slices = _DEFAULT_SIMULATION_TIME_SLICES

    if 'population_data_path' in config:
        population_data = PopulationData.import_population_data(config['population_data_path'])
    else:
        circle_consts = CirclesConsts(population_size=20_000)
        population_data = CirclesGenerator(circle_consts).population_data

    if 'matrix_data_path' in config:
        matrix_data = MatrixData.import_matrix_data(config['matrix_data_path'])
    else:
        matrix_consts = MatrixConsts()
        matrix_data = MatrixGenerator(population_data=population_data,
                                      matrix_consts=matrix_consts).matrix_data
    max_num_of_days = max(simulation_time_slices)

    consts = Consts(initial_infected_count=200,
                    change_policies=False,
                    partial_opening_active=False)
    sim_run_args = get_default_simulation_args_values()

    sm = SimulationManager((),
                           population_data,
                           matrix_data,
                           run_args=sim_run_args,
                           consts=consts)

    df_columns_new_infected_column = [f"new_infected_day_{i + 1}"
                                      for i in range(max_num_of_days)]
    df_columns_growth_rate_column = [f"infected_growth_day_{i + 1}"
                                     for i in range(1, max_num_of_days)]
    df = pd.DataFrame(columns=df_columns_new_infected_column + df_columns_growth_rate_column)
    total_loops = 0
    for number_of_days, number_of_loops in sorted(simulation_time_slices.items(), reverse=True):
        for i in range(total_loops, number_of_loops):
            sm.reset()
            sm.setup_sick()
            current_run_results = _compute_r0_for_one_run(sm, number_of_days)
            df.loc[i, :number_of_days] = current_run_results['new_sick_every_day']
            df.loc[i, max_num_of_days:
                      max_num_of_days+number_of_days-1] = current_run_results['growth_rates']
        total_loops += number_of_loops
    return df



if __name__ == "__main__":
    calculate_r_effective()
