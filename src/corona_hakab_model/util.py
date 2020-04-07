from abc import abstractmethod
from typing import Generic, List, TypeVar, Protocol

from scipy.stats import binom, randint, rv_discrete


def dist(*args):
    def const_dist(a):
        return rv_discrete(name="const", values=([a], [1]))()

    def uniform_dist(a, b):
        return randint(a, b + 1)

    def trig(a, c, b):
        # todo I have no idea what this distribution supposedly represents, we're gonna pretend it's
        #  an offset-binomial and call it a day

        return binom(b - a, (c - a) / (b - a), loc=a)

    if len(args) == 1:
        return const_dist(*args)
    if len(args) == 2:
        return uniform_dist(*args)
    if len(args) == 3:
        return trig(*args)
    raise TypeError


def upper_bound(d):
    return d.b + d.kwds.get("loc", 0)


def lower_bound(d):
    return d.a + d.kwds.get("loc", 0)


class HasDuration(Protocol):
    @abstractmethod
    def duration(self) -> int:
        pass


T = TypeVar("T", bound=HasDuration)


class Queue(Generic[T]):
    def __init__(self):
        self.queued: List[List[T]] = [[]]
        self.next_ind = 0

    def _resize(self, new_size):
        """
        resize the queue's internal list to support new_size-length durations
        """
        if new_size < len(self.queued):
            raise NotImplementedError
        new_array = []
        new_array.extend(self.queued[self.next_ind:])
        new_array.extend(self.queued[:self.next_ind])
        new_array.extend([[] for _ in range(new_size - len(self.queued))])

        self.queued = new_array
        self.next_ind = 0

    def append_at(self, element: T, duration: int):
        if duration >= len(self.queued):
            self._resize(duration+1)
        dest_ind = duration + self.next_ind
        if dest_ind >= len(self.queued):
            dest_ind -= len(self.queued)

        self.queued[dest_ind].append(element)

    def append(self, element: T):
        key = max(element.duration() - 1, 0)
        self.append_at(element, key)

    def extend(self, elements):
        for t in elements:
            self.append(t)

    def advance(self):
        ret = self.queued[self.next_ind]
        self.queued[self.next_ind] = []
        self.next_ind += 1
        if self.next_ind == len(self.queued):
            self.next_ind = 0
        return ret
