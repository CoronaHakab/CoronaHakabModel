from abc import ABC

from state_machine import State, StochasticState, TerminalState


class MedicalState(State, ABC):
    susceptible: bool
    contagiousness: float
    test_willingness: float
    detectable: bool

    def __init__(self, susceptible, contagiousness, test_willingness, detectable, **kwargs):
        self.susceptible = susceptible
        self.contagiousness = contagiousness
        self.test_willingness = test_willingness
        self.detectable = detectable
        super().__init__(**kwargs)

    @staticmethod
    def build(mechanism, susceptible, contagiousness, test_willingness, detectable, **kwargs):
        """
        Builds the appropriate MedicalState state object from a Dictionary
        @param detectable: bool, whether agent infection could theoretically be detected
        @param test_willingness: float, probability that agent will be willing to be tested
        @param contagiousness: float, probability of infecting others
        @param susceptible: bool, whether agent can be infected
        @param mechanism: str, Whether the medical state machine should be 'terminal' or 'stochastic'
        @param kwargs: dict, name parameter is required
        @return: an instance of a MedicalState child class
        """
        medical_state_type = mechanism

        if medical_state_type == 'terminal':
            return TerminalMedicalState(susceptible, contagiousness, test_willingness, detectable, **kwargs)
        elif medical_state_type == 'stochastic':
            return StochasticMedicalState(susceptible, contagiousness, test_willingness, detectable, **kwargs)
        else:
            raise KeyError("Medical state type invalid!")

    def val(self):
        return self.agent_count


class TerminalMedicalState(MedicalState, TerminalState):
    pass


class StochasticMedicalState(MedicalState, StochasticState):
    pass
