from collections import defaultdict
from typing import List

from common.agent import Agent
from common.medical_state_machine import MedicalStateMachine
from common.state_machine import PendingTransfers


class MedicalStateManager:
    """
    Manages the medical state
    """

    def __init__(self,
                 sim_manager: "manager.SimulationManager" = None,
                 medical_state_machine: MedicalStateMachine = None):
        """
        TODO: Find more effective way to give medical_state_machine when sim_manager is not needed
        :param sim_manager:  Defaults to None. The manager that we update.
                             If None, we dont update the manager and we use medical_state_machine
        :param medical_state_machine:
        """
        assert medical_state_machine or sim_manager,\
            "Manager and medical state machine cannot both be None"

        self.manager = sim_manager
        self.medical_state_machine = self.manager.medical_machine if self.manager \
            else medical_state_machine
        self.pending_transfers = PendingTransfers()

    def step(self, new_sick: List[Agent]):
        """
        :param new_sick: List of new agents that got sick
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
            if self.manager:
                agent.set_medical_state_no_inform(self.medical_state_machine.get_state_upon_infection(agent))
            else:  # TODO: Find a more elegant way to do this
                agent.medical_state = self.medical_state_machine.get_state_upon_infection(agent)

            changed_state_introduced[agent.medical_state].append(agent)

        # saves this number for supervising
        new_sick_counter = len(new_sick)

        moved = self.pending_transfers.advance()
        for (agent, destination, origin, _) in moved:
            if self.manager:
                agent.set_medical_state_no_inform(destination)
            else:  # TODO: Find a more elegant way to do this
                agent.medical_state = destination
            changed_state_introduced[destination].append(agent)
            changed_state_leaving[origin].append(agent)

        for state, agents in changed_state_introduced.items():
            state.add_many(agents)
            self.pending_transfers.extend(state.transfer(agents))

        for state, agents in changed_state_leaving.items():
            state.remove_many(agents)

        return dict(new_sick=new_sick_counter)
