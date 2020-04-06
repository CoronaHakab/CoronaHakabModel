from collections import defaultdict
from typing import Dict, Generic, List, Sequence, TypeVar, Iterable

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


def is_strict_sorted(s: Iterable):
    i = iter(s)
    try:
        prev = next(i)
    except StopIteration:
        return True
    for x in i:
        if x <= prev:
            return False
    return True


T = TypeVar("T")


class Queue(Generic[T]):
    def __init__(self):
        # in x time steps return the list of pending elements
        self.inner: Dict[int, List[T]] = defaultdict(list)

    def append(self, element: T):
        key = max(element.original_duration - 1, 0)
        self.inner[key].append(element)

    def extend(self, elements):
        for t in elements:
            self.append(t)

    def advance(self) -> Sequence[T]:
        # todo improve? (rotating array?)
        new_inner = defaultdict(list)
        ret = ()  # no elements to return in the current step
        for k, v in self.inner.items():
            if k > 0:
                new_inner[k - 1] = v
            else:
                ret = v  # the list of elements to return now (key=0)
        self.inner = new_inner
        return ret
