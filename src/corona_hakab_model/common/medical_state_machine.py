from .medical_state import MedicalState
from .state_machine import StateMachine


class MedicalStateMachine(StateMachine[MedicalState]):
    def __init__(self, initial_state: MedicalState, default_state_upon_infection: MedicalState, **kwargs):
        super().__init__(initial_state, **kwargs)
        self.default_state_upon_infection = default_state_upon_infection
        self.add_state(default_state_upon_infection)

    def get_state_upon_infection(self, agent) -> MedicalState:
        if agent:  # placeholder
            pass
        return self.default_state_upon_infection
        # todo add virtual link between initial and infected state for graphs
