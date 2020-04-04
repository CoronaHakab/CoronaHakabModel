from collections import defaultdict
from typing import List

import manager
from agent import Agent
from medical_state import MedicalState
from state_machine import PendingTransfer, PendingTransfers


class MedicalStateManager:
    """
   Manages the medical state
    """

    def __init__(self, sim_manager: "manager.SimulationManager"):
        self.manager = sim_manager
        self.pending_transfers = PendingTransfers()

    def step(self, new_sick: List[Agent]):
        # all the new sick agents are leaving their previous step
        changed_state_leaving = defaultdict(list)
        # agents which are going to enter the new state
        changed_state_introduced = defaultdict(list)
        # list of all the new sick agents

        for agent in new_sick:
            agent.set_medical_state_no_inform(self.manager.medical_machine.get_state_upon_infection(agent))

        # saves this number for supervising
        self.manager.new_sick_counter = len(new_sick)  # TODO should be handled in SimulationManager
        # all the new sick are going to get to the next state
        for agent in new_sick:
            changed_state_introduced[agent.medical_state].append(agent)

        moved = self.pending_transfers.advance()
        for (agent, destination, origin, _) in moved:
            agent.set_medical_state_no_inform(destination)

            changed_state_introduced[destination].append(agent)
            changed_state_leaving[origin].append(agent)

        for state, agents in changed_state_introduced.items():
            state.add_many(agents)
            self.pending_transfers.extend(state.transfer(agents))

        for state, agents in changed_state_leaving.items():
            state.remove_many(agents)

    def add_newly_infected(self, newly_infected):
        """
        Sets the default medical state to every newly infected agent
        """

    def advance_disease(self):
        moved = self.pending_transfers.advance()
        for (agent, destination, origin, _) in moved:
            agent.set_medical_state_no_inform(destination)

            changed_state_introduced[destination].append(agent)
            changed_state_leaving[origin].append(agent)

        for state, agents in changed_state_introduced.items():
            state.add_many(agents)
            self.pending_transfers.extend(state.transfer(agents))

        for state, agents in changed_state_leaving.items():
            state.remove_many(agents)
