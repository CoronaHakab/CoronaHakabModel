from manager import SimulationManager

if __name__ == "__main__":
    sm = SimulationManager(
        (
            "Recovered",
            "Deceased",
            "Symptomatic",
            "Asymptomatic",
            "Hospitalized",
            "ICU",
            "Latent",
            "Silent",
        )
    )
    sm.run()
    sm.plot(save=False, max_scale=False)
