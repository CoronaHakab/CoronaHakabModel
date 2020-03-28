from manager import SimulationManager
from supervisor import Supervisable, Supervisor
from consts import Consts
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
            Supervisable.Sum(
                "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized"
            )            
        )
   )
    sm.run()
    sm.plot(save=True, max_scale=False)


def compare_simulations_example():
    sm1 = SimulationManager(
        (Supervisable.Sum(
            "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", "Recovered", "Deceased"
        ),
         "Symptomatic",
         "Recovered"
        ), consts=Consts(r0=1.5)
    )
    sm1.run()

    sm2 = SimulationManager(
        (Supervisable.Sum(
            "Symptomatic", "Asymptomatic", "Latent", "Silent", "ICU", "Hospitalized", "Recovered", "Deceased"
        ),
         "Symptomatic",
         "Recovered"
        ), consts=Consts(r0=1.8)
    )
    sm2.run()

    Supervisor.static_plot(((sm1, f"ro = {sm1.consts.r0}:", ("y-", "y--", "y:")),
                            (sm2, f"ro = {sm2.consts.r0}:", ("c-", "c--", "c:"))),
                           f"comparing r0 = {sm1.consts.r0} to r0={sm2.consts.r0}")
