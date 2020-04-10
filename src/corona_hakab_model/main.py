from argparse import ArgumentParser

from consts import Consts
from generation.circles_consts import CirclesConsts
from generation.matrix_consts import MatrixConsts
from generation.generation_manager import GenerationManger
from manager import SimulationManager
from supervisor import LambdaValueSupervisable, Supervisable, Supervisor


def main():
    parser = ArgumentParser(
        """
    COVID-19 Simulation
    
    Two modes:
    1. Generation:
        python main.py generation [--circles-consts] [--matrix-consts]
    2. Full run:
        python main.py <Any other parameters>
    
    If ran in 'generation' mode, only '--matrix-consts' and '--circles-consts'
    are relevant.
    
    When running the simulation itself, either import the population OR give
    generation parameters.
    
    """
    )

    parser.add_argument("-s",
                        "--simulation-parameters",
                        dest="simulation_parameters_path",
                        help="Parameters for simulation engine")
    parser.add_argument("-c",
                        "--circles-consts",
                        dest="circles_consts_path",
                        help="Parameter file with consts for the circles")
    parser.add_argument("-m",
                        "--matrix-consts",
                        dest="matrix_consts_path",
                        help="Parameter file with consts for the matrix")
    args = parser.parse_args()

    if args.circles_consts_path:
        circles_consts = CirclesConsts.from_file(args.circles_consts_path)
    else:
        circles_consts = CirclesConsts()

    if args.matrix_consts_path:
        matrix_consts = MatrixConsts.from_file(args.matrix_consts_path)
    else:
        matrix_consts = MatrixConsts()

    gm = GenerationManger(circles_consts=circles_consts, matrix_consts=matrix_consts)

    if args.simulation_parameters_path:
        consts = Consts.from_file(args.simulation_parameters_path)
    else:
        consts = Consts()

    sm = SimulationManager(
        (
            # "Latent",
            # "Silent",
            # "Asymptomatic",
            # "Symptomatic",
            # "Deceased",
            # "Hospitalized",
            # "ICU",
            # "Susceptible",
            # "Recovered",
            Supervisable.Sum(
                "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", name="currently sick"
            ),
            # LambdaValueSupervisable("ever hospitalized", lambda manager: len(manager.medical_machine["Hospitalized"].ever_visited)),
            LambdaValueSupervisable(
                "was ever sick",
                lambda manager: len(manager.agents) - manager.medical_machine["Susceptible"].agent_count,
            ),
            # Supervisable.NewCasesCounter(),
            # Supervisable.GrowthFactor(
            #    Supervisable.Sum("Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized"),
            Supervisable.NewCasesCounter(),
            LambdaValueSupervisable("Detected Daily", lambda manager: manager.new_detected_daily),
            # LambdaValueSupervisable("Current Confirmed Cases", lambda manager: sum(manager.tested_positive_vector)),
            # Supervisable.R0(),
            # Supervisable.Delayed("Symptomatic", 3),
        ),
        gm.population_data,
        gm.matrix_data,
        consts=consts,
    )
    print(sm)
    sm.run()
    sm.plot(save=True, max_scale=False)


def compare_simulations_example():
    sm1 = SimulationManager(
        (
            Supervisable.Sum(
                "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", "Recovered", "Deceased"
            ),
            "Symptomatic",
            "Recovered",
        ),
        consts=Consts(r0=1.5),
    )
    sm1.run()

    sm2 = SimulationManager(
        (
            Supervisable.Sum(
                "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", "Recovered", "Deceased"
            ),
            "Symptomatic",
            "Recovered",
        ),
        consts=Consts(r0=1.8),
    )
    sm2.run()

    Supervisor.static_plot(
        ((sm1, f"ro = {sm1.consts.r0}:", ("y-", "y--", "y:")), (sm2, f"ro = {sm2.consts.r0}:", ("c-", "c--", "c:"))),
        f"comparing r0 = {sm1.consts.r0} to r0={sm2.consts.r0}",
    )


if __name__ == "__main__":
    main()
