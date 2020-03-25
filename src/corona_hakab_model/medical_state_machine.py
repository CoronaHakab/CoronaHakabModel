from medical_state import MedicalState
from state_machine import StateMachine


class MedicalStateMachine(StateMachine[MedicalState]):
    def __init__(self, initial_state: MedicalState, state_upon_infection: MedicalState, **kwargs):
        super().__init__(initial_state, **kwargs)
        self.state_upon_infection = state_upon_infection
        self.add_state(state_upon_infection)

        # todo add virtual link between initial and infected state for graphs
