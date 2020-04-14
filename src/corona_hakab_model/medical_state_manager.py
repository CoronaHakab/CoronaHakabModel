from collections import defaultdict
from typing import List

from agent import Agent
from medical_state_machine import MedicalStateMachine
from state_machine import PendingTransfers


class MedicalStateManager:
    """
    Manages the medical state
    """

    def __init__(self, medical_state_machine: MedicalStateMachine):
        self.medical_state_machine = medical_state_machine
        self.pending_transfers = PendingTransfers()

    def step(self, new_sick: List[Agent], update_only_medical_state: bool = False):
        """

        :param new_sick: List of new agents that got sick
        :param update_only_medical_state: Defaults to False. If True, update agent.medical_state directly.
                                          If False it also updates agent.manager.
        :return:
        """
        # all the new sick agents are leaving their previous step
        changed_state_leaving = defaultdict(list)
        # agents which are going to enter the new state
        changed_state_introduced = defaultdict(list)
        # list of all the new sick agents

        # all the new sick are going to get to the next state
        for agent in new_sick:
            changed_state_leaving[agent.medical_state].append(agent)
            if update_only_medical_state:
                agent.medical_state = self.medical_state_machine.get_state_upon_infection(agent)
            else:
                agent.set_medical_state_no_inform(self.medical_state_machine.get_state_upon_infection(agent))
            changed_state_introduced[agent.medical_state].append(agent)

        # saves this number for supervising
        new_sick_counter = len(new_sick)

        moved = self.pending_transfers.advance()
        for (agent, destination, origin, _) in moved:
            if update_only_medical_state:
                agent.medical_state = destination
            else:
                agent.set_medical_state_no_inform(destination)
            changed_state_introduced[destination].append(agent)
            changed_state_leaving[origin].append(agent)

        for state, agents in changed_state_introduced.items():
            state.add_many(agents)
            self.pending_transfers.extend(state.transfer(agents))

        for state, agents in changed_state_leaving.items():
            state.remove_many(agents)

        return dict(new_sick=new_sick_counter)
