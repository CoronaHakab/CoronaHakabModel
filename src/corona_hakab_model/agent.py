class Agent:
    """
    This class represents a person in our doomed world.
    """
    __slots__ = "index", "home", "work", "medical_state", "is_home_quarantined", "is_full_quarantined", "manager"

    def __init__(self, index, manager, initial_state):
        self.index = index
        self.home = None
        self.work = None

        self.medical_state = None
        self.manager = manager

        self.is_home_quarantined = False
        self.is_full_quarantined = False

        self.set_medical_state(initial_state)

    def set_medical_state(self, new_state):
        if self.medical_state:
            self.medical_state.remove_agent(self)
        new_state.add_agent(self)
        self.medical_state = new_state

        self.manager.infectiousness_vector[self.index] = new_state.infectiousness

    def __str__(self):
        return "<Person,  index={}, medical={}>".format(self.index, self.medical_state)

    def get_infection_ratio(self):
        return self.medical_state.infectousness

    def add_home(self, home):
        self.home = home

    def add_work(self, work):
        self.work = work

    def change_medical_state(self, new_status):
        self.medical_state = new_status


class Circle:
    __slots__ = "kind", "agents"

    def __init__(self, kind: str):
        self.kind = kind
        self.agents = set()

    def add_agent(self, agent):
        self.agents.add(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)

    def get_indexes_of_my_circle(self, my_index):
        rest_of_circle = {o.index for o in self.agents}
        rest_of_circle.remove(my_index)
        return rest_of_circle
