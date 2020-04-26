from abc import abstractmethod
from collections import OrderedDict
from typing import Generic, List, Protocol, TypeVar

import numpy as np


def dist(*args):
    def const_dist(a):
        return get_numpy_uniform_dist(a=a)

    def uniform_dist(a, b):
        return get_numpy_uniform_dist(a=a, b=b)

    def off_binom(a, c, b):
        # todo I have no idea what this distribution supposedly represents, we're gonna pretend it's
        #  an offset-binomial whose mean is c and call it a day
        return lambda size=None: np.random.binomial(n=b - a,
                                                    p=(c - a) / (b - a),
                                                    size=size) + a

    if len(args) == 1:
        return const_dist(*args)
    if len(args) == 2:
        return uniform_dist(*args)
    if len(args) == 3:
        return off_binom(*args)
    raise TypeError


def upper_bound(d):
    return d.b + d.kwds.get("loc", 0)


def get_numpy_uniform_dist(a, b=None):
    """
    :param a: lower bound of an interval
    :param b: upper bound of the interval. If b=None than we have discrete distribution.
    :return: Return a distribution that gets number of elements to sample. If b=None this is a discrete distribution.
             O/w this is uniform distribution over [a,b]
    """
    if b:
        range_to_choose_from = list(range(a, b))
    else:
        range_to_choose_from = [a]
    return lambda size=None: np.random.choice(range_to_choose_from, size=size)


def parse_str_to_num(val):
    try:
        return int(val)
    except ValueError:
        return float(val)


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
        new_array.extend(self.queued[: self.next_ind])
        new_array.extend([[] for _ in range(new_size - len(self.queued))])

        self.queued = new_array
        self.next_ind = 0

    def append_at(self, element: T, duration: int):
        if duration >= len(self.queued):
            self._resize(duration + 1)
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


class BucketDict(OrderedDict):
    """
    This class supports missing values if there is a key larger than requested.
    for example: dict is {1:1,5:5,10:10}
    then:
     x<=1 return 1
     1<x<=5 return 5
     5<x<=10 return 10
     10<x return 10
    """

    def __missing__(self, key):
        if not self.keys():
            return 0

        for dict_key in self.keys():
            if key <= dict_key:
                return self[dict_key]
        return self[dict_key]

    @property
    def mean_val(self):
        # TODO maybe need to consider age distribution in mean calculation. or wait until monte carlo normalize..
        return np.array(list(self.values())).mean() if self.keys() else 0
