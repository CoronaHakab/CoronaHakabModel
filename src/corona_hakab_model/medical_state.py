from abc import ABC

from state_machine import State, StochasticState, TerminalState


class MedicalState(State, ABC):
    susceptible: bool
    contagiousness: float
    test_willingness: float
    detectable: bool

    def __init__(self, **kwargs):
        self.susceptible = kwargs.get('susceptible')
        self.contagiousness = kwargs.get('contagiousness')
        self.test_willingness = kwargs.get('test_willingness')
        self.detectable = kwargs.get('detectable')
        super().__init__(kwargs.get('name'))

    @staticmethod
    def build(**kwargs):
        """
        Builds the appropriate MedicalState state object from a Dictionary
        @param kwargs: dictionary of MedicalState parameters and additionally 'mechanism' which defines how the state
        should function
        @return: an instance of a MedicalState child class
        """
        medical_state_type = kwargs.pop('mechanism')
        new_kwargs = {k: kwargs.get(k, None) for k in ("name",
                                                       "susceptible",
                                                       "contagiousness",
                                                       "test_willingness",
                                                       "detectable")}

        if medical_state_type == 'terminal':
            return TerminalMedicalState(**new_kwargs)
        elif medical_state_type == 'stochastic':
            return StochasticMedicalState(**new_kwargs)

    def val(self):
        return self.agent_count


class TerminalMedicalState(MedicalState, TerminalState):
    pass


class StochasticMedicalState(MedicalState, StochasticState):
    pass
