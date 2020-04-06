from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medical_state import MedicalState
    from manager import SimulationManager


class Agent:
    """
    This class represents a person in our doomed world.
    """

    __slots__ = (
        "index",
        "medical_state",
        "manager",
        "age",
        "geographic_circle",
        "social_circles",
    )

    # todo note that this changed to fit generation. should update simulation manager accordingly
    def __init__(self, index):
        self.index = index

        self.geographic_circle = None
        self.social_circles = []

        # don't know if this is necessary
        self.manager: SimulationManager = None
        self.medical_state: MedicalState = None

    def add_to_simulation(self, manager: SimulationManager, initial_state: MedicalState):
        self.manager = manager
        self.set_medical_state_no_inform(initial_state)

    def set_test_start(self):
        self.manager.date_of_last_test[self.index] = self.manager.current_step

    def set_test_result(self, test_result):
        # TODO: add a property here
        self.manager.tested_positive_vector[self.index] = test_result
        self.manager.tested_vector[self.index] = True

    def set_medical_state_no_inform(self, new_state: MedicalState):
        self.medical_state = new_state
        self.manager.contagiousness_vector[self.index] = new_state.contagiousness

        if new_state == self.manager.medical_machine.states_by_name["Deceased"]:
            self.manager.living_agents_vector[self.index] = False

        self.manager.susceptible_vector[self.index] = new_state.susceptible
        self.manager.test_willingness_vector[self.index] = new_state.test_willingness

    def __str__(self):
        return f"<Person,  index={self.index}, medical={self.medical_state}>"

    def get_infection_ratio(self):
        return self.medical_state.contagiousness


class Circle:
    __slots__ = "kind", "agent_count"

    def __init__(self):
        self.agent_count = 0

    def add_many(self, agents):
        self.agent_count += len(agents)

    def remove_many(self, agents):
        self.agent_count -= len(agents)

    def add_agent(self, agent):
        self.agent_count += 1

    def remove_agent(self, agent):
        self.agent_count -= 1


class TrackingCircle(Circle):
    __slots__ = ("agents",)

    def __init__(self):
        super().__init__()
        self.agents = set()

    def add_agent(self, agent):
        super().add_agent(agent)
        if agent in self.agents:
            raise ValueError("DuplicateAgent")
        self.agents.add(agent)
        assert self.agent_count == len(self.agents)

    def remove_agent(self, agent):
        super().remove_agent(agent)
        self.agents.remove(agent)
        assert self.agent_count == len(self.agents)

    def add_many(self, agents):
        super().add_many(agents)
        if self.agents.intersection(set(agents)):
            raise ValueError("DuplicateAgent")
        self.agents.update(agents)
        assert self.agent_count == len(
            self.agents
        ), f"self.agent_count: {self.agent_count}, len(self.agents): {len(self.agents)}"

    def remove_many(self, agents):
        super().remove_many(agents)
        self.agents.difference_update(agents)
        assert self.agent_count == len(self.agents)

    def get_indexes_of_my_circle(self, my_index):
        rest_of_circle = {o.index for o in self.agents}
        rest_of_circle.remove(my_index)
        return rest_of_circle
