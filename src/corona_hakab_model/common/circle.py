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
    __slots__ = ("agents", "ever_visited")

    def __init__(self):
        super().__init__()
        self.agents = set()
        # done to count how many different agents ever visited a given state
        self.ever_visited = set()

    def add_agent(self, agent):
        super().add_agent(agent)
        if agent in self.agents:
            raise ValueError("DuplicateAgent")
        self.agents.add(agent)
        self.ever_visited.add(agent)
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
        self.ever_visited.update(agents)
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
