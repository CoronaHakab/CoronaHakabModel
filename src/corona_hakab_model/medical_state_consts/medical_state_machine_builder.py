import json
import numbers
from typing import Dict, List, TypedDict, Union

from medical_state import StochasticMedicalState, MedicalState
from medical_state_machine import MedicalStateMachine
from state_machine import State
from util import dist

# This file includes various tools to load a JSON file to a MedicalStateMachine.
# The data classes here are used to type the JSON objects after they are parsed to dictionaries.
# Note: JSON should be validated with the medical-state-machine-schema.json JSON schema to ensure that the data is valid


class ProbabilityData(TypedDict):
    type: str
    from_state: str
    to: str


class MedicalStateData(TypedDict):
    susceptible: bool
    contagiousness: float
    test_willingness: float
    detectable: bool
    name: str
    mechanism: str


class StateTransitionData(TypedDict):
    from_state: str
    to: str
    duration_distribution: List[float]
    probability: Union[float, ProbabilityData]


class MedicalStateMachineData(TypedDict):
    medical_states: Dict[str, MedicalStateData]
    initial_state: str
    default_state_upon_infection: str
    medical_state_transitions: List[StateTransitionData]
    sick_states: List[str]
    was_ever_sick_states: List[str]


class MedicalStateMachineBuilder:
    __slots__ = "data"

    def __init__(self, param_path):
        """
        Load parameters from JSON file to a local data field on this class
        """
        with open(param_path, "rt") as read_file:
            self.data: MedicalStateMachineData = json.loads(read_file.read())

    def create_state_machine(self):
        """
        Creates a MedicalStateMachine object from the data loaded from the JSON file, validating the data along the way
        """
        medical_states_dict = self.data['medical_states']
        medical_states: Dict[str, MedicalState] = {}
        for name, medical_state in medical_states_dict.items():
            medical_states[name] = MedicalState.build(**medical_state, name=name)

        initial_state = medical_states.get(self.data.get('initial_state'))
        default_state_upon_infection = medical_states.get(self.data.get('default_state_upon_infection'))
        sick_states = self.data.get('sick_states')
        was_ever_sick_states = self.data.get('was_ever_sick_states')

        if initial_state and default_state_upon_infection:
            return_value = MedicalStateMachine(initial_state, default_state_upon_infection, sick_states, was_ever_sick_states)
        else:
            raise ValueError("Either initial state or default state upon infection not defined correctly!")

        for transition in self.data['medical_state_transitions']:
            from_state = medical_states.get(transition.get('from_state'))
            to_state = medical_states.get(transition.get('to'))
            distribution = ('duration_distribution' in transition and
                            1 <= len(transition['duration_distribution']) <= 3 and
                            dist(*transition.get('duration_distribution')))

            if isinstance(from_state, StochasticMedicalState) and isinstance(to_state, State) and distribution:
                if 'probability' in transition:
                    probability = transition.get('probability')
                    if isinstance(probability, numbers.Number):
                        from_state.add_transfer(to_state, distribution, probability)
                    else:
                        if probability.get('type') == 'allEventualitiesExcept':
                            match = next(d for d in self.data['medical_state_transitions'] if
                                         d.get('from_state') == probability.get('from_state') and
                                         d.get('to') == probability.get('to'))
                            if isinstance(match.get('probability'), numbers.Number):
                                from_state.add_transfer(to_state, distribution, 1 - match['probability'])
                            else:
                                raise ValueError(
                                    f"Probability not defined for {match.get('from_state')} to {match.get('to')}")
                        else:
                            raise ValueError(f"Probability invalid for {from_state} to {to_state}")
                else:
                    from_state.add_transfer(to_state, distribution, ...)
            else:
                raise ValueError(f"Medical state transition [{transition}] not defined correctly!")

        return return_value
