from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import (
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from agent import Agent, Circle
from scipy.stats import rv_discrete
from util import upper_bound

PendingTransfer = namedtuple(
    "PendingTransfer", ["agent", "target_state", "origin_state", "original_duration"]
)

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

    def __str__(self):
        return f"<state {self.name}>"


class StochasticState(State):
    # todo enforce probabilities sum to 1?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probs_cumulative = np.array([], dtype=float)
        self.destinations: List[State] = []
        self.durations: List[rv_discrete] = []

    def prob_specific(self, ind):
        if ind == 0:
            return self.probs_cumulative[0]
        return self.probs_cumulative[ind] - self.probs_cumulative[ind - 1]

    def add_transfer(
        self,
        destination: State,
        duration: rv_discrete,
        probability: Union[float, type(...)],
    ):
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
        transfer_indices = np.searchsorted(
            self.probs_cumulative, np.random.random(len(agents))
        )
        bin_count = np.bincount(transfer_indices)
        if len(bin_count) > len(self.probs_cumulative):
            raise Exception("probs must sum to 1")
        durations = [
            iter(d.rvs(c))
            for (c, s, d) in zip(bin_count, self.destinations, self.durations)
        ]
        return [
            PendingTransfer(
                agent,
                self.destinations[transfer_ind],
                self,
                durations[transfer_ind].__next__(),
            )
            for transfer_ind, agent in zip(transfer_indices, agents)
        ]


class TerminalState(State):
    def transfer(self, agents: Set[Agent]) -> Iterable[PendingTransfer]:
        return ()


T = TypeVar("T", bound=State)


class StateMachine(Generic[T]):
    def __init__(self, initial_state: T):
        self.initial: T = initial_state

        self.states_by_name = {initial_state.name: initial_state}
        self.state_indices = {initial_state: 0}
        self.states = [initial_state]

    def __getitem__(self, item: Union[str, Tuple[str, ...]]):
        if isinstance(item, str):
            return self.states_by_name[item]
        return tuple(self[i] for i in item)

    def add_state(self, state: T):
        if self.states_by_name.setdefault(state.name, state) is not state:
            raise Exception(f"duplicate state name {state.name}")
        if state not in self.state_indices:
            self.state_indices[state] = len(self.state_indices)
            self.states.append(state)

        state.machine = self

    @cached_property
    def markovian(self):
        """
        Return a markovian matrix for all the states (inc. partial transfers)
        returns a 3-tuple: the matrix, the indices for the terminal nodes, and the entry columns for all states
        """

        # each initial transfer state gets its own index
        transfer_starts: Dict[StochasticState, Dict[int, int]] = {}

        # terminal_states each get their own index
        terminal_states: Dict[TerminalState, int] = {}

        next_index = 0
        for s in self.states:
            if isinstance(s, TerminalState):
                terminal_states[s] = next_index
                next_index += 1
            elif isinstance(s, StochasticState):
                state_dict = {}
                for i, dur in enumerate(s.durations):
                    state_dict[i] = next_index
                    next_index += upper_bound(dur)
                transfer_starts[s] = state_dict
            else:
                raise TypeError

        ret = np.zeros((next_index, next_index), dtype=float)
        entry_columns: Dict[State, np.ndarray] = {}
        for t_state, i in terminal_states.items():
            ec = np.zeros(next_index, float)
            ec[i] = 1
            entry_columns[t_state] = ec

        for s_state, s_dict in transfer_starts.items():
            ec = np.zeros(next_index, float)
            for transfer_index, node_index in s_dict.items():
                p_of_transfer = s_state.prob_specific(transfer_index)
                ec[node_index] = p_of_transfer
            entry_columns[s_state] = ec

        for t_state, terminal_index in terminal_states.items():
            # todo if this is slow we can do it faster
            ret[:, terminal_index] = entry_columns[t_state]

        for s_state, sub_dict in transfer_starts.items():
            for transfer_index, transfer_start in sub_dict.items():
                dur = s_state.durations[transfer_index]
                dest = s_state.destinations[transfer_index]
                p_passed = 1
                for i in range(upper_bound(dur) + 1):
                    p_exit_unbiased = dur.pmf(i)
                    p_exit = p_exit_unbiased / p_passed
                    p_passed *= 1 - p_exit
                    ret[:, transfer_start + i] = p_exit * entry_columns[dest]
                    ret[transfer_start + i + 1, transfer_start + i] += 1 - p_exit

        # todo remove?
        for col in range(next_index):
            assert np.sum(ret[:, col]) == 1

        return ret, terminal_states, entry_columns

    # todo function to draw a pretty graph
