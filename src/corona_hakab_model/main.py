from manager import SimulationManager
from supervisor import Supervisable

if __name__ == "__main__":
    sm = SimulationManager(
        (
            Supervisable.Stack(
            "Symptomatic",
            "Deceased",
            "Asymptomatic",
            "Hospitalized",
            "ICU",
            "Latent",
            "Silent",
            "Susceptible",
            "Recovered"),
        )
    )
    sm.run()
    sm.stackplot()
