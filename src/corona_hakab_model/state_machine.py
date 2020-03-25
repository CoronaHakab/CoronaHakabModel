from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
from math import fsum
from typing import List, Dict, Set, Optional, Iterable, Collection, Union, Tuple, Sequence, Generic, TypeVar, Iterator

import numpy as np
from scipy.stats import rv_discrete

from agent import Circle, Agent
from util import upper_bound, lower_bound

PendingTransfer = namedtuple("PendingTransfer", ["agent", "target_state", "origin_state", "original_duration"])

TransferCollection = Dict[int, List[PendingTransfer]]


class PendingTransfers:
    def __init__(self):
        self.inner: Dict[int, List[PendingTransfer]] = defaultdict(list)

    def append(self, transfer: PendingTransfer):
        key = transfer.original_duration
        self.inner[key].append(transfer)

    def extend(self, transfers):
        for t in transfers:
            self.append(t)

    def advance(self) -> Sequence[PendingTransfer]:
        # todo improve? (rotating array?)
        new_inner = defaultdict(list)
        ret = ()
        for k, v in self.inner.items():
            if k:
                new_inner[k - 1] = v
            else:
                ret = v
        self.inner = new_inner
        return ret


class State(Circle, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.machine: Optional[StateMachine] = None

    def _add_descendant(self, child: State):
        self.machine.add_state(child)

    @abstractmethod
    def transfer(self, agents: Collection[Agent]) -> Iterable[PendingTransfer]:
        pass

    @abstractmethod
    def probability(self, time: int, state: State, tol: float) -> float:
        pass

    def __str__(self):
        return f"<state {self.name}>"


class StochasticState(State):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs_cumulative = np.array([], dtype=float)
        self.destinations: List[State] = []
        self.durations: List[rv_discrete] = []

    def probs_specific(self):
        ret = np.copy(self.probs_cumulative)
        ret[1:] -= ret[:1]
        return ret

    def add_transfer(self, destination: State, duration: rv_discrete, probability: Union[float, type(...)]):
        if probability is ...:
            p = 1
        elif self.durations:
            p = self.probs_cumulative[-1] + probability
            if p > 1:
                raise ValueError("probability higher than 1")
        else:
            p = probability
        # todo improve?
        self.probs_cumulative = np.append(self.probs_cumulative, p)

        self.destinations.append(destination)
        self.durations.append(duration)

        self._add_descendant(destination)

    def transfer(self, agents: Set[Agent]) -> Iterable[PendingTransfer]:
        transfer_indices = np.searchsorted(self.probs_cumulative, np.random.random(len(agents)))
        bin_count = np.bincount(transfer_indices)
        durations = [
            iter(d.rvs(c)) for (c, s, d) in zip(bin_count, self.destinations, self.durations)
        ]
        return [
            PendingTransfer(agent, self.destinations[transfer_ind], self, durations[transfer_ind].__next__())
            for transfer_ind, agent in zip(transfer_indices, agents)
        ]

    def probability(self, time: int, state: State, tol: float) -> float:
        if state is self:
            return fsum(
                (probability * duration.sf(time))
                for probability, duration in zip(self.probs_specific(), self.durations)
            )
        ret = 0
        for probability, duration, dest in zip(self.probs_specific(), self.durations, self.destinations):
            if probability <= tol:
                continue
            for trans_time in range(lower_bound(duration), upper_bound(duration) + 1):
                if time - trans_time < 0:
                    break
                stp = probability * duration.pmf(trans_time)
                if stp > tol:
                    ret += (stp * dest.probability(time - trans_time, state, tol / stp))
        return ret


class TerminalState(State):
    def transfer(self, agents: Set[Agent]) -> Iterable[PendingTransfer]:
        return ()

    def probability(self, time: int, state: State, tol: float) -> float:
        return state is self


T = TypeVar('T', bound=State)


class StateMachine(Generic[T]):
    def __init__(self, initial_state: T):
        self.initial: T = initial_state

        self.states_by_name = {initial_state.name: initial_state}
        self.state_indices = {initial_state: 0}
        self.states = [initial_state]

    def __getitem__(self, item: Union[str, Tuple[str, ...]]):
        if isinstance(item, str):
            return self.states_by_name[item]
        return (self[i] for i in item)

    def add_state(self, state: T):
        if self.states_by_name.setdefault(state.name, state) is not state:
            raise Exception(f"duplicate state name {state.name}")
        self.state_indices[state] = len(self.state_indices)
        self.states.append(state)

        state.machine = self

    # todo function to draw a pretty graph
