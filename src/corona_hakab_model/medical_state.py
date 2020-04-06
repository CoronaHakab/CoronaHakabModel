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

    def __init__(self, *args, test_willingness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_willingness = test_willingness

    susceptible = True
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

    def __init__(self, *args, detectable: bool, test_willingness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_willingness = test_willingness
        self.detectable = detectable
