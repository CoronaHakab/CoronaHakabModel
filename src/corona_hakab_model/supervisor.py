# flake8: noqa flake8 doesn't support named expressions := so for now we have to exclude this file for now:(

from __future__ import annotations

from datetime import datetime
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, List, NamedTuple, Sequence, Union, Dict

from project_structure import SIM_OUTPUT_FOLDER
from pathlib import Path
import manager
from state_machine import StochasticState

import numpy as np
import pandas as pd
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state_machine import State

from histogram import TimeHistograms

class SimulationProgression:
    """
    records statistics about the simulation.
    """

    # todo I want the supervisor to decide when the simulation ends
    # todo record write/read results as text

    def __init__(self, supervisables: Sequence[Supervisable], manager: "manager.SimulationManager"):
        self.supervisables = supervisables
        self.manager = manager

        self.time_vector = []

    def snapshot(self, manager):
        t = self.manager.current_step
        self.time_vector.append(t)

        # Assert the first snapshot is done at t=0 & there is a single snapshot at each time step
        assert self.time_vector[t] == t

        for s in self.supervisables:
            s.snapshot(manager)

    def dump(self, filename=None):
        file_name = Path(filename) if filename else SIM_OUTPUT_FOLDER / ("final_results.csv")
        file_name.parent.mkdir(parents=True, exist_ok=True)

        tabular_supervisables = [s for s in self.supervisables if isinstance(s, TabularSupervisable)]
        value_supervisables = [s for s in self.supervisables if isinstance(s, ValueSupervisable)]

        # Output each tabular sample to a new csv file
        # (all samples of a given supervisable are gathered in a new subfolder)
        for s in tabular_supervisables:
            day_to_table_dict = s.publish()
            for day, table in day_to_table_dict.items():
                sample_file_name = SIM_OUTPUT_FOLDER / f"{s.name()} {day}.csv"
                df = pd.DataFrame(table)
                df.to_csv(sample_file_name)

        all_data = dict([s.publish() for s in value_supervisables])

        df = pd.DataFrame(all_data, index=self.time_vector)
        df.to_csv(file_name)
        return df


class Supervisable(ABC):
    @abstractmethod
    def snapshot(self, manager: "manager.SimulationManager"):
        pass

    @abstractmethod
    def publish(self):
        pass

    # todo is_finished
    # todo supervisables should be able to keep the manager running if they want

    @abstractmethod
    def name(self) -> str:
        pass

    @classmethod
    @lru_cache
    def coerce(cls, arg, manager: "manager.SimulationManager") -> Supervisable:
        if isinstance(arg, str):
            return _StateSupervisable(manager.medical_machine[arg])
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, Callable):
            return arg(manager)
        raise TypeError

    class State:

        class Current:
            def __init__(self, name) -> None:
                self.name = name

            def __call__(self, m):
                return _StateSupervisable(m.medical_machine[self.name])

        class TotalSoFar:
            def __init__(self, name) -> None:
                self.name = name

            def __call__(self, m):
                return _StateTotalSoFarSupervisable(m.medical_machine[self.name])

        class AddedPerDay:
            def __init__(self, name) -> None:
                self.name = name

            def __call__(self, m):
                diff_sup = _DiffSupervisable(_StateTotalSoFarSupervisable(m.medical_machine[self.name]))
                return _NameOverrideSupervisable(diff_sup, "New " + self.name)

    class Wrappers:
        class RunningAverage:
            def __init__(self, supervisable, window_size) -> None:
                self.supervisable = supervisable
                self.window_size = window_size
                self.func = lambda arr: np.convolve(
                    np.concatenate([np.zeros(window_size - 1), arr]),
                    np.ones(window_size) / window_size,
                    'valid'
                )

            def __call__(self, m):
                sup = Supervisable.coerce(self.supervisable, m)
                return _PostProcessSupervisor(sup, self.func,
                                              sup.name() + f' - {self.window_size} days running average')

        class Growth:
            def __init__(self, supervisable, num_of_days_to_group_together=1) -> None:
                self.supervisable = supervisable
                self.num_of_days_to_group_together = num_of_days_to_group_together

                def foo(arr):
                    cumsum = arr.cumsum()
                    res = np.zeros_like(arr, dtype=float) + np.nan
                    for i in range(num_of_days_to_group_together, len(arr)):
                        if cumsum[i - num_of_days_to_group_together] == 0:
                            continue

                        res[i] = (cumsum[i] - cumsum[i - num_of_days_to_group_together]) / cumsum[
                            i - num_of_days_to_group_together]

                    return res

                self.func = foo

            def __call__(self, m):
                sup = Supervisable.coerce(self.supervisable, m)
                return _PostProcessSupervisor(sup, self.func,
                                              sup.name() + f' - {self.num_of_days_to_group_together} days growth')

    class Delayed(NamedTuple):
        arg: Any
        delay: int

        def __call__(self, m):
            return _DelayedSupervisable(Supervisable.coerce(self.arg, m), self.delay)

    class Diff(NamedTuple):
        arg: Any

        def __call__(self, m):
            return _DiffSupervisable(Supervisable.coerce(self.arg, m))

    class Stack:
        def __init__(self, *args):
            self.args = args

        def __call__(self, m):
            return _StackedFloatSupervisable([Supervisable.coerce(a, m) for a in self.args])

    class Sum:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, m):
            return _SumSupervisable([Supervisable.coerce(a, m) for a in self.args], **self.kwargs)

    class R0:
        def __init__(self):
            pass

        def __call__(self, m):
            return _EffectiveR0Supervisable()

    class NewCasesCounter:
        def __init__(self):
            pass

        def __call__(self, manager):
            return _NewInfectedCount()

    class CurrentInfectedTable:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, manager):
            return _CurrentInfectedTable(*self.args, **self.kwargs)

    class GrowthFactor:
        def __init__(self, sum_supervisor: "Sum", new_infected_supervisor: "NewCasesCounter"):
            self.new_infected_supervisor = new_infected_supervisor
            self.sum_supervisor = sum_supervisor

        def __call__(self, m):
            return _GrowthFactor(
                Supervisable.coerce(self.new_infected_supervisor, m), Supervisable.coerce(self.sum_supervisor, m)
            )

    class TimeBasedHistograms:
        def __init__(self):
            pass

        def __call__(self, manager):
            return _TimeBasedHistograms()

    class SupervisiblesLambda(NamedTuple):
        supervisiables: Sequence
        func: Callable
        name: str

        def __call__(self, m):
            return _PostProcessSupervisor([Supervisable.coerce(s, m) for s in self.supervisiables], self.func,
                                          self.name)



SupervisableMaker = Callable[[Any], Supervisable]


class ValueSupervisable(Supervisable):
    def __init__(self):
        self.data = []

    @abstractmethod
    def get(self, manager: "manager.SimulationManager"):
        pass

    def snapshot(self, manager: "manager.SimulationManager"):
        self.data.append(self.get(manager))

    def publish(self):
        return self.name(), np.array(self.data)


class TabularSupervisable(Supervisable):
    def __init__(self, interval: int):
        self.data = []
        self.sampling_days = []
        self.interval = interval

    def names(self):
        return [f"t0 +{d} days" for d in self.sampling_days]

    @abstractmethod
    def get(self, manager: "manager.SimulationManager"):
        pass

    def snapshot(self, manager: "manager.SimulationManager"):
        if manager.current_step % self.interval == 0:
            self.data.append(self.get(manager))
            self.sampling_days.append(manager.current_step)

    def publish(self):
        return {name_i: data_i for name_i, data_i in zip(self.names(), self.data)}


class LambdaValueSupervisable(ValueSupervisable):
    def __init__(self, name: str, lam: Callable):
        super().__init__()
        self._name = name
        self.lam = lam

    def name(self) -> str:
        return self._name

    def get(self, manager) -> float:
        return self.lam(manager)


class _StateSupervisable(ValueSupervisable):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def get(self, manager: "manager.SimulationManager") -> float:
        return self.state.agent_count

    def name(self) -> str:
        return self.state.name


class _StateTotalSoFarSupervisable(ValueSupervisable):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        self.__name = state.name + " So Far"

    def get(self, manager: "manager.SimulationManager") -> float:
        return len(self.state.ever_visited)

    def name(self) -> str:
        return self.__name


class _CurrentInfectedTable(TabularSupervisable):
    def __init__(self, interval):
        super().__init__(interval)
        self.sick_states = None

    def get(self, manager) -> Dict[str, List]:
        if self.sick_states is None:
            medical_states = manager.medical_machine.states_by_name.values()
            self.sick_states = [s for s in medical_states if isinstance(s, StochasticState)]

        agent_ids = []
        medical_status = []

        for state in self.sick_states:
            agent_ids += [agent.index for agent in state.agents]
            medical_status += [state.name] * state.agent_count
        return {
            "agent_id" : agent_ids,
            "medical_status" : medical_status
        }

    def name(self) -> str:
        return "infected_table"


class _DelayedSupervisable(ValueSupervisable):
    def __init__(self, inner: ValueSupervisable, delay: int):
        super().__init__()
        self.inner = inner
        self.delay = delay

    def get(self, manager: "manager.SimulationManager") -> float:
        desired_date = manager.current_step - self.delay
        if desired_date < 0:
            return np.nan
        else:
            return self.inner.data[desired_date]

    def name(self) -> str:
        return self.inner.name() + f" + {self.delay} days"

    def names(self):
        return [n + f" + {self.delay} days" for n in self.inner.names()]

    def snapshot(self, manager: "manager.SimulationManager"):
        self.inner.snapshot(manager)
        super().snapshot(manager)


class _NameOverrideSupervisable(ValueSupervisable):
    def __init__(self, inner: ValueSupervisable, name: str):
        super().__init__()
        self.__name = name
        self.inner = inner

    def get(self, manager: "manager.SimulationManager"):
        return self.inner.get(manager)

    def snapshot(self, manager: "manager.SimulationManager"):
        self.inner.snapshot(manager)
        super().snapshot(manager)

    def publish(self):
        return super().publish()

    def name(self) -> str:
        return self.__name


class _DiffSupervisable(ValueSupervisable):
    def __init__(self, inner: ValueSupervisable):
        super().__init__()
        self.inner = inner

    def get(self, manager: "manager.SimulationManager") -> float:
        if manager.current_step < 1:
            return 0
        return self.inner.data[manager.current_step] - self.inner.data[manager.current_step - 1]

    def snapshot(self, manager: "manager.SimulationManager"):
        self.inner.snapshot(manager)
        super().snapshot(manager)

    def name(self) -> str:
        return self.inner.name() + " diff"


class VectorSupervisable(ValueSupervisable, ABC):
    @abstractmethod
    def names(self):
        pass

    def _to_ys(self):
        raise NotImplementedError('Need to convert (.x and .y) notation to the new (.data) notation')
        n = len(self.y[0])
        return [[v[i] for v in self.y] for i in range(n)]

    def publish(self):
        raise NotImplementedError('Need to convert (.x and .y) notation to the new (.data) notation')
        return [["", self.names()]] + ([[z[0]] + z[1] for z in zip(self.data.items())])


class _StackedFloatSupervisable(VectorSupervisable):
    def __init__(self, inners: List[ValueSupervisable]):
        super().__init__()
        self.inners = inners

    def get(self, manager: "manager.SimulationManager"):
        return [i.get(manager) for i in self.inners]

    def name(self) -> str:
        return "Stacked (" + ", ".join(n.name() for n in self.inners) + ")"

    def names(self):
        return [i.name() for i in self.inners]


class _SumSupervisable(ValueSupervisable):
    def __init__(self, inners: List[ValueSupervisable], **kwargs):
        super().__init__()
        self.inners = inners
        self.kwargs = kwargs

    def get(self, manager: "manager.SimulationManager") -> float:
        return sum(s.get(manager) for s in self.inners)

    def names(self):
        return ["Total(" + ", ".join(names) + ")" for names in zip(*(i.names() for i in self.inners))]

    def name(self) -> str:
        if "name" in self.kwargs:
            return self.kwargs["name"]
        return "Total(" + ", ".join(n.name() for n in self.inners)


# todo this is broken. needs adaptation to parasymbolic matrix
class _EffectiveR0Supervisable(ValueSupervisable):
    def __init__(self):
        super().__init__()

    def get(self, manager) -> float:
        # note that this calculation is VARY heavy
        suseptable_indexes = np.flatnonzero(manager.susceptible_vector)
        # todo someone who knows how this works fix it
        return (
                np.sum(1 - np.exp(manager.matrix[suseptable_indexes].data))
                * manager.update_matrix_manager.total_contagious_probability
                / len(manager.agents)
        )

    def name(self) -> str:
        return "effective R"


class _NewInfectedCount(ValueSupervisable):
    def __init__(self):
        super().__init__()

    def get(self, manager) -> float:
        return manager.new_sick_counter

    def name(self) -> str:
        return "new infected"


class _GrowthFactor(ValueSupervisable):
    def __init__(self, new_infected_supervisor, sum_supervisor):
        super().__init__()
        self.new_infected_supervisor = new_infected_supervisor
        self.sum_supervisor = sum_supervisor

    def get(self, manager) -> float:
        new_infected = self.new_infected_supervisor.get(manager)
        sum = self.sum_supervisor.get(manager)
        if sum == 0:
            return np.nan
        return new_infected / sum

    def name(self) -> str:
        return "growth factor"


class _TimeBasedHistograms(Supervisable):
    def __init__(self):
        self._name = 'time-based histograms'
        self.histograms = TimeHistograms()

    def name(self):
        return self._name

    def snapshot(self, manager: "manager.SimulationManager"):
        matrix = manager.matrix.get_scipy_sparse()
        self.histograms.update_all_histograms(matrix)

    def publish(self):
        return {self.name(): self.histograms.get()}


class _PostProcessSupervisor(ValueSupervisable):
    def __init__(self, supervisables: Union[List[Supervisable], Supervisable], func: Callable, name: str):
        super().__init__()
        self._name = name
        self.func = func
        if isinstance(supervisables, Supervisable):
            supervisables = [supervisables]
        self.supervisables = supervisables

    def get(self, manager: "manager.SimulationManager"):
        pass

    def snapshot(self, manager: "manager.SimulationManager"):
        [v.snapshot(manager) for v in self.supervisables]
        # super().snapshot(manager)

    def publish(self):
        vectors = [s.publish()[1] for s in self.supervisables]
        expected_length_of_result = len(vectors[0])

        res = self.func(*vectors)

        # If the function shortens the array length (e.g. diff()), pad with zeros
        if len(res) < expected_length_of_result:
            temp = np.zeros_like(vectors[0])
            temp[-len(res):] = res
            res = temp

        self.data = res
        return super().publish()

    def name(self) -> str:
        return self._name
