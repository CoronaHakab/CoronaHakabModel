from dataclasses import dataclass


@dataclass
class InfectionRatios:
    symptomatic: float = 0.75
    asymptomatic: float = 0.25
    silent: float = 0.3
