from manager import SimulationManager
from supervisor import Supervisable

if __name__ == "__main__":
    sm = SimulationManager(
        (
            "Symptomatic",
            Supervisable.Delayed("Symptomatic", 3),
            "Deceased",
            "Asymptomatic",
            "Hospitalized",
            "ICU",
            "Latent",
            "Silent",
            "Susceptible",
            "Recovered",
            Supervisable.Sum("Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", "Recovered", "Deceased"),
        )
    )
    sm.run()
    sm.plot(save=True, max_scale=False)
