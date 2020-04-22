from random import choice


class Node:
    __slots__ = "index", "connected"

    def __init__(self, index):
        self.index = index
        self.connected = set()

    def add_connections(self, new):
        if isinstance(new, list):
            self.connected.update(new)
        else:
            self.connected.add(new)

    def pop_random(self) -> "Node":
        to_pop = choice(list(self.connected))
        self.connected.remove(to_pop)
        return to_pop

    @staticmethod
    def connect(first: "Node", second: "Node"):
        first.add_connections(second)
        second.add_connections(first)
