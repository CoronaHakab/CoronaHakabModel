from __future__ import annotations

from argparse import ArgumentParser
import matplotlib_set_backend
import matplotlib.pyplot as plt

import numpy as np

from bsa.universal import write
from consts import Consts
from corona_hakab_model_data.__data__ import __version__
from generation.circles_consts import CirclesConsts
from generation.generation_manager import GenerationManger
from generation.matrix_consts import MatrixConsts
from manager import SimulationManager
from supervisor import LambdaValueSupervisable, Supervisable, SimulationProgression



from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

def main():
    parser = ArgumentParser("COVID-19 Simulation")

    sub_parsers = parser.add_subparsers(dest="sub_command")
    gen = sub_parsers.add_parser("generate", help="only generate the population data without running the simulation")
    gen.add_argument("output")

    parser.add_argument(
        "-s", "--simulation-parameters", dest="simulation_parameters_path", help="Parameters for simulation engine"
    )
    parser.add_argument(
        "-c", "--circles-consts", dest="circles_consts_path", help="Parameter file with consts for the circles"
    )
    parser.add_argument(
        "-m", "--matrix-consts", dest="matrix_consts_path", help="Parameter file with consts for the matrix"
    )
    parser.add_argument("--version", action="version", version=__version__)
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

    if args.sub_command == "generate":
        with open(args.output, "wb") as w:
            write(gm.matrix_data.matrix, w)
        return

    if args.simulation_parameters_path:
        consts = Consts.from_file(args.simulation_parameters_path)
    else:
        consts = Consts()

    sm = SimulationManager(
        (
            # "Latent",
            Supervisable.State.AddedPerDay("Asymptomatic"),
            Supervisable.State.Current("Asymptomatic"),
            Supervisable.State.TotalSoFar("Asymptomatic"),
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
            Supervisable.NewCasesCounter(),
            Supervisable.Wrappers.Growth(Supervisable.NewCasesCounter(), 1),
            Supervisable.Wrappers.RunningAverage(Supervisable.Wrappers.Growth(Supervisable.NewCasesCounter()), 7),
            Supervisable.Wrappers.Growth(Supervisable.NewCasesCounter(), 7),
            # Supervisable.GrowthFactor(
            #    Supervisable.Sum("Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized"),
            # LambdaValueSupervisable("Detected Daily", lambda manager: manager.new_detected_daily),
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
    df: pd.DataFrame = sm.dump()
    df.plot()
    plt.show()




if __name__ == "__main__":
    main()
