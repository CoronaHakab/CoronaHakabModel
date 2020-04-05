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
        self.manager = None
        self.medical_state = None

    def add_to_simulation(self, manager, initial_state: "medical_state.MedicalState"):
        self.manager = manager
        self.set_medical_state_no_inform(initial_state)

    def set_medical_state_no_inform(self, new_state: "medical_state.MedicalState"):
        self.medical_state = new_state
        # count how many entered silent state
        if new_state == self.manager.medical_machine.states_by_name["Silent"]:
            self.manager.in_silent_state += 1
        self.manager.contagiousness_vector[self.index] = new_state.contagiousness
        self.manager.susceptible_vector[self.index] = new_state.susceptible

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
        self.agents.add(agent)
        assert self.agent_count == len(self.agents)

    def remove_agent(self, agent):
        super().remove_agent(agent)
        self.agents.remove(agent)
        assert self.agent_count == len(self.agents)

    def add_many(self, agents):
        super().add_many(agents)
        self.agents.update(agents)
        assert self.agent_count == len(self.agents)

    def remove_many(self, agents):
        super().remove_many(agents)
        self.agents.difference_update(agents)
        assert self.agent_count == len(self.agents)

    def get_indexes_of_my_circle(self, my_index):
        rest_of_circle = {o.index for o in self.agents}
        rest_of_circle.remove(my_index)
        return rest_of_circle
