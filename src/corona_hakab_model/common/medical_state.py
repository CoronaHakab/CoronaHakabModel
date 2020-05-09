from abc import ABC

from .state_machine import State  # , StochasticState, TerminalState
from .util import BucketDict


class MedicalState(State, ABC):
    susceptible: bool
    contagiousness: BucketDict
    test_willingness: float
    detectable: bool

    def val(self):
        return self.agent_count


class SusceptibleState(MedicalState, ABC):
    def __init__(self, *args, test_willingness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_willingness = test_willingness

        self.contagiousness = BucketDict({0:[0]})

    susceptible = True
    detectable = False


class ContagiousState(MedicalState, ABC):
    susceptible = False

    def __init__(self, *args, detectable: bool, contagiousness: BucketDict, test_willingness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.detectable = detectable
        self.test_willingness = test_willingness
        self.contagiousness = contagiousness


class ImmuneState(MedicalState, ABC):
    susceptible = False

    def __init__(self, *args, detectable: bool, test_willingness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_willingness = test_willingness
        self.detectable = detectable
        self.contagiousness = BucketDict({0:[0]})
