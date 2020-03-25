from manager import SimulationManager
from supervisor import Supervisor, Delayed

if __name__ == "__main__":
    sm = SimulationManager(
        # ("Recovered", "Deceased", "Symptomatic", "Asymptomatic", "Hospitalized", "ICU", "Latent", "Silent")
        ("Symptomatic", Delayed("Symptomatic", 3), "Deceased", "Asymptomatic", "Hospitalized", "ICU", "Latent", "Silent")
    )
    sm.run()
    sm.plot(save=True, max_scale=False)
