from abc import ABC

from state_machine import State


class MedicalState(State, ABC):
    susceptible: bool
    contagiousness: float

    def val(self):
        return self.agent_count


class SusceptibleState(MedicalState, ABC):
    susceptible = True
    contagiousness = 0


class ContagiousState(MedicalState, ABC):
    susceptible = False

    def __init__(self, *args, contagiousness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.contagiousness = contagiousness


class ImmuneState(MedicalState, ABC):
    susceptible = False
    contagiousness = 0
