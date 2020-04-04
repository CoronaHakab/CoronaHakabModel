import medical_state


class Agent:
    """
    This class represents a person in our doomed world.
    """

    __slots__ = (
        "index",
        "medical_state",
        "is_home_isolated",
        "is_full_isolated",
        "simulation_manager",
        "traits"
    )

    def __init__(self, index, simulation_manager, initial_state: "medical_state.MedicalState"):
        self.index = index

        self.traits = AgentTraits()

        self.simulation_manager = simulation_manager

        self.medical_state: "medical_state.MedicalState" = None
        self.set_medical_state_no_inform(initial_state)

        self.is_home_isolated = False
        self.is_full_isolated = False

    def set_medical_state_no_inform(self, new_state: "medical_state.MedicalState"):
        self.medical_state = new_state
        # count how many entered silent state
        if new_state == self.simulation_manager.medical_machine.states_by_name["Silent"]:
            self.simulation_manager.in_silent_state += 1
        self.simulation_manager.contagiousness_vector[self.index] = new_state.contagiousness
        self.simulation_manager.susceptible_vector[self.index] = new_state.susceptible

    def __str__(self):
        return f"<Person,  index={self.index}, medical={self.medical_state}>"

    def get_infection_ratio(self):
        return self.medical_state.contagiousness


class AgentTraits:
    """
    Individual traits of an agent - Age, Susceptability, obedience...
    """
    def __init__(self, age: int = 25):
        self.age = age


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
