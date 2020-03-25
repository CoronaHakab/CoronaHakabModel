from abc import ABC

from state_machine import State  # , StochasticState, TerminalState


class MedicalState(State, ABC):
    infectable: bool
    infectiousness: float

    def val(self):
        return self.agent_count


class InfectableState(MedicalState, ABC):
    infectable = True
    infectiousness = 0


class InfectiousState(MedicalState, ABC):
    infectable = False

    def __init__(self, *args, infectiousness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.infectiousness = infectiousness


class ImmuneState(MedicalState, ABC):
    infectable = False
    infectiousness = 0
