from abc import ABC
from collections import defaultdict

from state_machine import State  # , StochasticState, TerminalState
from typing import Callable, Dict,List

class MedicalState(State, ABC):
    susceptible: bool
    contagiousness: float

    def val(self):
        return self.agent_count

class SusceptibleState(MedicalState, ABC):
    susceptible = True
    contagiousness = 0


class ContagiousState(MedicalState, ABC):
    susceptible = False

    def __init__(self, *args, contagiousness: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.contagiousness = contagiousness


class ImmuneState(MedicalState, ABC):
    susceptible = False
    contagiousness = 0

class MedicalStateManager:
    """
   Manages the medical state
    """

    def __init__(self, sim_manager: "manager.SimulationManager"):
        self.manager =sim_manager
        
    def step(self, new_sick: Dict[MedicalState, List]):
        # all the new sick agents are leaving their previous step
        changed_state_leaving = new_sick
        # agents which are going to enter the new state
        changed_state_introduced = defaultdict(list)
        # list of all the new sick agents
        new_sick_list = sum(changed_state_leaving.values(), [])

        # saves this number for supervising
        self.manager.new_sick_counter = len(new_sick_list)
        # all the new sick are going to get to the next state
        changed_state_introduced[self.manager.medical_machine.state_upon_infection] = new_sick_list

        for s in new_sick_list:
            s.set_medical_state_no_inform(self.manager.medical_machine.state_upon_infection)

        moved = self.manager.pending_transfers.advance()
        for (agent, destination, origin, _) in moved:
            agent.set_medical_state_no_inform(destination)

            changed_state_introduced[destination].append(agent)
            changed_state_leaving[origin].append(agent)

        for state, agents in changed_state_introduced.items():
            state.add_many(agents)
            self.manager.pending_transfers.extend(state.transfer(agents))

        for state, agents in changed_state_leaving.items():
            state.remove_many(agents)