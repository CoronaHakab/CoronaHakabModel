import json
import numbers
from typing import Dict, List, TypedDict, Union, Optional

from errors import InvalidJSON
from medical_state import StochasticMedicalState, TerminalMedicalState, MedicalState
from medical_state_machine import MedicalStateMachine
from state_machine import State
from util import dist


class ProbabilityData(TypedDict):
    type: str
    from_state: str
    to: str


class MedicalStateData(TypedDict, total=False):
    susceptible: bool
    contagiousness: float
    test_willingness: float
    detectable: bool
    name: str
    stochastic: Optional[bool]
    terminal: Optional[bool]


class StateTransitionData(TypedDict):
    from_state: str
    to: str
    distribution: List[float]
    probability: Union[float, ProbabilityData]


class MedicalStateMachineData(TypedDict):
    medical_states: Dict[str, MedicalStateData]
    initial_state: str
    default_state_upon_infection: str
    medical_state_transitions: List[StateTransitionData]
    sick_states: List[str]
    was_ever_sick_states: List[str]


class MedicalStateMachineCreator:
    __slots__ = "data"

    def __init__(self, param_path):
        """
        Load parameters from JSON file
        """
        with open(param_path, "rt") as read_file:
            self.data: MedicalStateMachineData = json.loads(read_file.read())

    def create_state_machine(self):
        """
        Creates a MedicalStateMachine object from the data loaded from the JSON file
        """
        medical_states = self.data['medical_states']
        medical_state_machines: Dict[str, MedicalState] = {}
        for name, medical_state in medical_states.items():
            medical_state['name'] = name
            if 'terminal' in medical_state and medical_state.pop('terminal'):
                state_machine = TerminalMedicalState(**medical_state)
                if medical_state.get('stochastic'):
                    raise InvalidJSON(f"Medical state [{name}] cannot be both stochastic and terminal!")
            elif 'stochastic' in medical_state and medical_state.pop('stochastic'):
                state_machine = StochasticMedicalState(**medical_state)
            else:
                raise InvalidJSON(f"Medical state [{name}: {medical_state}] neither stochastic nor terminal!")
            medical_state_machines[name] = state_machine

        initial_state = medical_state_machines.get(self.data.get('initial_state'))
        default_state_upon_infection = medical_state_machines.get(self.data.get('default_state_upon_infection'))
        sick_states = self.data.get('sick_states')
        was_ever_sick_states = self.data.get('was_ever_sick_states')

        if initial_state and default_state_upon_infection:
            ret = MedicalStateMachine(initial_state, default_state_upon_infection, sick_states, was_ever_sick_states)
        else:
            raise InvalidJSON("Either initial state or default state upon infection not defined correctly!")

        for transition in self.data['medical_state_transitions']:
            from_state = medical_state_machines.get(transition.get('from_state'))
            to_state = medical_state_machines.get(transition.get('to'))
            distribution = 'distribution' in transition and 1 <= len(transition['distribution']) <= 3 and \
                           dist(*transition.get('distribution'))

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
                                raise InvalidJSON(
                                    f"Probability not defined for {match.get('from_state')} to {match.get('to')}")
                        else:
                            raise InvalidJSON(f"Probability invalid for {from_state} to {to_state}")
                else:
                    from_state.add_transfer(to_state, distribution, ...)
            else:
                raise InvalidJSON(f"Medical state transition [{transition}] not defined correctly!")

        return ret
