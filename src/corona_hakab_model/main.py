from manager import SimulationManager

if __name__ == "__main__":
    sm = SimulationManager(
        # ("Recovered", "Deceased", "Symptomatic", "Asymptomatic", "Hospitalized", "ICU", "Latent", "Silent")
        ("Symptomatic", "Recovered", "Deceased", "Asymptomatic", "Hospitalized", "ICU", "Latent", "Silent",
         "Susceptible")
    )
    sm.run()
    sm.plot(save=True, max_scale=False)
