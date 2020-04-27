from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Collection, Dict, Generic, Iterable, List, NamedTuple, Optional, Set, Tuple, TypeVar, Union

import numpy as np
from agent import Agent, TrackingCircle
from util import Queue


class PendingTransfer(NamedTuple):
    agent: Agent
    target_state: State
    origin_state: State
    original_duration: int

    def duration(self) -> int:
        return self.original_duration


TransferCollection = Dict[int, List[PendingTransfer]]


class PendingTransfers(Queue[PendingTransfer]):
    pass


class StochasticTransferGenerator:
    """
    This is the actual class that handles stochastic progression -
    It had a set of probabilities for a set of outcomes - both the next state and the duration of the current state.

    """

    # todo enforce probabilities sum to 1?
    def __init__(self):
        self.probs_cumulative: np.ndarray = np.array([], dtype=float)
        self.destinations: List[State] = []
        self.durations = []

    def prob_specific(self, ind: int) -> float:
        return self.probs_cumulative[ind]

    def add_transfer(self, destination: State, duration, probability: Union[float, type(...)]):
        if probability is ...:
            p = 1 - np.sum(self.probs_cumulative)
        else:
            p = probability
        # todo improve?
        self.probs_cumulative = np.append(self.probs_cumulative, p)
        if np.sum(self.probs_cumulative) > 1:
            raise ValueError("Probabilities sum to more than 1")

        self.destinations.append(destination)
        self.durations.append(duration)

    def transfer(self, agents: Set[Agent], origin_state: State) -> Iterable[PendingTransfer]:

        new_states_list = np.random.choice(a=len(self.destinations),
                                           size=len(agents),
                                           p=self.probs_cumulative)
        pending_transfers = []
        # for each agent, create the pending transfer of the predetermined outcome.
        for agent_index, agent in enumerate(agents):
            agent_new_state_ind = new_states_list[agent_index]
            duration_by_age_dist = self.durations[agent_new_state_ind][agent.age]()
            pending_transfers.append(PendingTransfer(agent,
                                                     self.destinations[agent_new_state_ind],
                                                     origin_state,
                                                     duration_by_age_dist))
        return pending_transfers


class State(TrackingCircle, ABC):
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
        self.generator = StochasticTransferGenerator()

    def add_transfer(self, destination: State, duration: BucketDict, probability: Union[float, type(...)]):
        self.generator.add_transfer(destination, duration, probability)

        self._add_descendant(destination)

    def transfer(self, agents: Set[Agent]) -> Iterable[PendingTransfer]:
        return self.generator.transfer(agents, self)

    def prob_specific(self, ind: int) -> float:
        return self.generator.prob_specific(ind)

    @property
    def durations(self):
        return self.generator.durations

    @property
    def destinations(self):
        return self.generator.destinations


class AgentAwareState(State):
    """
    An agent aware state is one that sorts agents into buckets, and holds a specific generator for each bucket.
    Actions are performed on each bucket.

    At the moment, only age is accounted for
    TODO account for more parameters
    """

    # todo enforce probabilities sum to 1?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This dict will hold the different generators of the state, and will use the appropriate one based on traits
        # For now, this only takes note of age, meaning the dict is {age_range: StochasticTransferGenerator}
        self.generators = defaultdict(StochasticTransferGenerator)
        self.agents_by_bucket = defaultdict(set)
        self.sorted_buckets = []

    def get_bucket_for_transfer(self, max_age):
        """
        Get bucket to add Transfer into. adding transfers CAN create new buckets, but we need a way to create a
        bucket for a set of parameters.

        right now we only have max_age
        """
        if max_age in sorted(self.generators):  # Placeholder
            pass
        return max_age

    def get_bucket_for_agent(self, agent):
        """
        Fit agent into bucket.
        For now, buckets are the max age range.
        """
        for bucket in self.sorted_buckets:
            if agent.age < bucket:
                return bucket
        raise ValueError(f"Agent doesnt have handler for state {self.__str__()}")

    def add_transfer(self, max_age, destination: State, duration: rv_discrete, probability: Union[float, type(...)]):
        """
        Add a new transfer to the correct bucket.

        TODO max_age is only temporary, it should be possible to be more specific
        """
        assert self.agent_count == 0, "Cannot add new transfers after agents entered the state!"
        bucket = self.get_bucket_for_transfer(max_age)
        self.generators[bucket].add_transfer(destination, duration, probability)
        self.sorted_buckets = sorted(self.generators)

        self._add_descendant(destination)

    def prob_specific(self, ind: int) -> float:
        """
        Weighted avarage of the transfers
        """
        # TODO - support destinations average for normalizing?
        raise NotImplementedError
        if len(self.sorted_buckets) == 1:
            return self.generators[self.sorted_buckets[0]].prob_specific(ind)

    @property
    def durations(self):
        # TODO - support durations average for normalizing?
        raise NotImplementedError
        if len(self.sorted_buckets) == 1:
            return self.generators[self.sorted_buckets[0]].durations

    @property
    def destinations(self):
        # TODO - support destinations average for normalizing?
        raise NotImplementedError
        if len(self.sorted_buckets) == 1:
            return self.generators[self.sorted_buckets[0]].destinations

    def transfer(self, agents: Set[Agent]) -> Iterable[PendingTransfer]:
        """
        Run transfer on each bucket and return sum of PendingTransfer lists
        """
        pending_transfers = []

        # Sort each agent into the correct bucket
        incoming_agents_by_bucket = defaultdict(set)
        for agent in agents:
            incoming_agents_by_bucket[self.get_bucket_for_agent(agent)].add(agent)

        # transfer each agents group using the correct generator
        for bucket, agents in incoming_agents_by_bucket.items():
            # Extend list with each bucket's transfers
            pending_transfers.extend(self.generators[bucket].transfer(agents, self))

        return pending_transfers

    def validate_agents_count(self):
        """
        Make sure the sum of all agent counts in all buckets is the same as the total agents count
        """
        agents_count_in_buckets = 0
        for agents in self.agents_by_bucket.values():
            agents_count_in_buckets += len(agents)
        return agents_count_in_buckets == self.agent_count and self.agent_count == len(self.agents)

    def add_agent(self, agent: Agent):
        """
        Update agents_by_bucket - single agent
        """
        super().add_agent(agent)
        bucket = self.get_bucket_for_agent(agent)

        # Sanity
        assert agent not in self.agents_by_bucket[bucket], "Adding the same Agent into the same state twice!"

        self.agents_by_bucket[bucket].add(agent)
        # Sanity of counters
        assert self.validate_agents_count()

    def add_many(self, agents: Iterable[Agent]):
        """
        Update agents_by_bucket - slightly faster than updating each agent
        """
        super().add_many(agents)
        added_agents_by_bucket = defaultdict(set)
        # Sort agents into buckets
        for agent in agents:
            added_agents_by_bucket[self.get_bucket_for_agent(agent)].add(agent)
        for bucket, agents in added_agents_by_bucket.items():
            # Sanity
            assert not self.agents_by_bucket[bucket].intersection(
                agents
            ), "Adding the same Agent into the same state twice!"

            self.agents_by_bucket[bucket].update(agents)

        # Sanity of counters
        assert self.validate_agents_count()

    def remove_agent(self, agent):
        """
        Opposite of add_agent
        """
        super().remove_agent(agent)
        self.agents_by_bucket[self.get_bucket_for_agent(agent)].remove(agent)

        # Sanity of counters
        assert self.validate_agents_count()

    def remove_many(self, agents):
        """
        Opposite on add_many
        """
        super().remove_many(agents)
        removed_agents_by_bucket = defaultdict(set)
        for agent in agents:
            removed_agents_by_bucket[self.get_bucket_for_agent(agent)].add(agent)
        for bucket, agents in removed_agents_by_bucket.items():
            self.agents_by_bucket[bucket].difference_update(agents)

        # Sanity of counters
        assert self.validate_agents_count()


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

    # todo function to draw a pretty graph
