import random
from abc import abstractmethod
from collections import OrderedDict
from typing import Generic, List, Protocol, TypeVar, Callable
from functools import partial
import numpy as np
import scipy.stats as stats


class DiscreteDistribution(object):
    def __init__(self, domain:List[int], probs: List[float], prf: Callable):
        self.domain = domain
        self.probs = probs
        self.prf = prf

    def __call__(self, size=None):
        return self.prf(size)

    def prob(self, day):
        if day not in self.domain:
            return 0
        day_index = self.domain.index(day)
        return self.probs[day_index]


def dist(*args):
    def const_dist(a):
        domain = [a]
        probs = [1.]
        prf = partial(lambda size=None: np.random.choice(domain, size=size))
        return DiscreteDistribution(domain, probs, prf)

    def uniform_dist(a, b):
        domain = list(range(a, b+1))
        probs = [1. / len(domain)] * len(domain)
        prf = partial(lambda size=None: np.random.choice(domain, size=size))
        return DiscreteDistribution(domain, probs, prf)

    def weighted_dist(elements, p):
        prf = partial(lambda size=None: random.choices(elements, weights=p, k=size))
        return DiscreteDistribution(elements, p, prf)

    def off_binom(a, c, b):
        # todo I have no idea what this distribution supposedly represents, we're gonna pretend it's
        #  an offset-binomial whose mean is c and call it a day
        n = b-a
        p = (c-a)/(b-a)
        domain = list(range(a, b + 1))

        binom_rv = stats.binom(n, p, loc = a) # Calculates "Binom(n, p) + a"
        probs = [binom_rv.pmf(day) for day in domain]

        prf = partial(lambda size=None: np.random.binomial(n=n,
                                                            p=p,
                                                            size=size) + a)
        return DiscreteDistribution(domain, probs, prf)

    if len(args) == 1:
        return const_dist(*args)
    if len(args) == 2:
        if type(args[0]) == list and type(args[1]) == list:
            assert len(args[0]) == len(args[1]), f"Elements and weights vectors should be in same size! (elements: {len(args[0])}, weights: {len(args[1])})"
            return weighted_dist(*args)
        else:
            return uniform_dist(*args)
    if len(args) == 3:
        return off_binom(*args)
    raise TypeError


def parse_str_to_num(val):
    try:
        return int(val)
    except ValueError:
        return float(val)


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


K = TypeVar("K")
V = TypeVar("K")


class BucketDict(OrderedDict, Generic[K, V]):
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
