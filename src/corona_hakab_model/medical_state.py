from abc import ABC

from state_machine import State, StochasticState, TerminalState


class MedicalState(State, ABC):
    susceptible: bool
    contagiousness: float
    test_willingness: float
    detectable: bool

    def __init__(self, *args, **kwargs):
        self.susceptible = kwargs.pop('susceptible')
        self.contagiousness = kwargs.pop('contagiousness')
        self.test_willingness = kwargs.pop('test_willingness')
        self.detectable = kwargs.pop('detectable')
        super().__init__(*args, **kwargs)

    def val(self):
        return self.agent_count


class TerminalMedicalState(MedicalState, TerminalState):
    pass


class StochasticMedicalState(MedicalState, StochasticState):
    pass
