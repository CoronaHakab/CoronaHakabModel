import math


class Node:
    __slots__ = "index", "connected"

    def __init__(self, index):
        self.index = index
        self.connected = []

    def add_connections(self, new):
        try:
            self.connected.extend(new)
        except TypeError:
            self.connected.append(new)

    def pop_random(self, rand: float) -> "Node":
        rand = rand * len(self.connected)
        index = math.floor(rand)
        return self.connected.pop(index)

    @staticmethod
    def connect(first: "Node", second: "Node"):
        first.add_connections(second)
        second.add_connections(first)
