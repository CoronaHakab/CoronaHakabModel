from common.util import HasDuration, Queue


def test_queue():
    class Element(HasDuration):
        def __init__(self, v, d):
            self.val = v
            self.dur = d

        def duration(self) -> int:
            return self.dur

        def __repr__(self):
            return f"Element({self.val})"

    def advance():
        return frozenset(e.val for e in queue.advance())

    queue = Queue()
    queue.extend([Element("a_0", 1), Element("b_0", 2), Element("d_0", 4), Element("c_0", 3)])
    queue.append(Element("a_1", 1))

    assert advance() == {"a_0", "a_1"}
    queue.extend([Element("b_1", 1), Element("e_0", 4)])
    assert advance() == {"b_0", "b_1"}
    assert advance() == {"c_0"}
    assert advance() == {"d_0"}
    assert advance() == {"e_0"}
    assert advance() == set()
    queue.append(Element("f_0", 5))
    for _ in range(4):
        assert advance() == set()
    assert advance() == {"f_0"}


if __name__ == "__main__":
    test_queue()
