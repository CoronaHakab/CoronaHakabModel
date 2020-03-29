from scipy.stats import rv_discrete
from agent import Agent
from typing import List


class CircularConnectionsMatrix:

    __slots__ = "type", "agents", "circle_size_probability", "connection_strength"

    # todo switch type to a named-tuple
    def __init__(self, type: str, agents: List[Agent], circle_size_probability: rv_discrete, connection_strength: float):
        self.type = type
        self.agents = agents
        self.circle_size_probability = circle_size_probability
        self. connection_strength = connection_strength

class NonCircularConnectionMatrix:

    __slots__ = "type", "agents", "scale_factor", "connection_strength"

    def __init__(self, type: str, agents: List[Agent], scale_factor: float, connection_strength: float):
        self.type = type
        self.agents = agents
        self.scale_factor = scale_factor
        self.connection_strength = connection_strength

