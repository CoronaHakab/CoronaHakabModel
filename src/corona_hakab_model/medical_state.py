from abc import ABC

from state_machine import State  # , StochasticState, TerminalState


class MedicalState(State, ABC):
    susceptible: bool
    contagiousness: float
    test_willingness: float
    detectable: bool

    def val(self):
        return self.agent_count


class SusceptibleState(MedicalState, ABC):
    susceptible = True
    contagiousness = 0
    test_willingness = 0
    detectable = False


class ContagiousState(MedicalState, ABC):
    susceptible = False
    detectable = True

    def __init__(self, *args, contagiousness: float, test_willingness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_willingness = test_willingness
        self.contagiousness = contagiousness


class ImmuneState(MedicalState, ABC):
    susceptible = False
    contagiousness = 0
    test_willingness = 0

    def __init__(self, *args, detectable, **kwargs):
        super().__init__(*args, **kwargs)
        self.detectable = detectable
