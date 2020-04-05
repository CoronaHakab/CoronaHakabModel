import math
from random import sample


class Node:
    __slots__ = "index", "connected"

    def __init__(self, index):
        self.index = index
        self.connected = set()

    def add_connections(self, new):
        try:
            self.connected.add(new)
        except TypeError:
            self.connected.update(new)

    def pop_random(self) -> "Node":
        to_pop = sample(self.connected, 1)[0]
        self.connected.remove(to_pop)
        return to_pop

    @staticmethod
    def connect(first: "Node", second: "Node"):
        first.add_connections(second)
        second.add_connections(first)
