from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect
from functools import lru_cache
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, List

import numpy as np
from state_machine import TerminalState

try:
    import PySide2
except ImportError:
    pass
else:
    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use("Qt5Agg")
        del matplotlib
    del PySide2

try:
    # plt is optional
    from matplotlib import pyplot as plt
except ImportError:
    pass


class Supervisor:
    """
    records and plots statistics about the simulation.
    """

    # todo I want the supervisor to decide when the simulation ends
    # todo record write/read results as text

    def __init__(self, supervisables: Sequence[Supervisable], manager):
        self.supervisables = supervisables
        self.manager = manager

    def snapshot(self, manager):
        for s in self.supervisables:
            s.snapshot(manager)

    # todo stacked_plot

    def plot(self, max_scale=True, auto_show=True, save=True):
        output_dir = "../output/"
        total_size = self.manager.consts.population_size
        title = f"Infections vs. Days, size={total_size:,}"
        if max_scale:
            height = total_size
        else:
            height = max(s[1] for a in self.supervisables if (s := a.scale()))
        text_height = height / 2

        fig, ax = plt.subplots()

        # policies
        if self.manager.consts.active_quarantine:
            title = title + "\napplying lockdown from day {} to day {}".format(
                self.manager.consts.stop_work_days, self.manager.consts.resume_work_days
            )
            ax.axvline(x=self.manager.consts.stop_work_days, color="#0000ff")
            ax.text(
                self.manager.consts.stop_work_days + 2,
                text_height,
                f"day {self.manager.consts.stop_work_days} - pause all work",
                rotation=90,
            )
            ax.axvline(x=self.manager.consts.resume_work_days, color="#0000cc")
            ax.text(
                self.manager.consts.resume_work_days + 2,
                text_height,
                f"day {self.manager.consts.resume_work_days} - resume all work",
                rotation=90,
            )
        if self.manager.consts.home_quarantine_sicks:
            title = (
                title
                + "\napplying home quarantine for confirmed cases ({} of cases)".format(
                    self.manager.consts.caught_sicks_ratio
                )
            )
        if self.manager.consts.full_quarantine_sicks:
            title = (
                title
                + "\napplying full quarantine for confirmed cases ({} of cases)".format(
                    self.manager.consts.caught_sicks_ratio
                )
            )

        # plot parameters
        ax.set_title(title)
        ax.set_xlabel("days", color="#1C2833")
        ax.set_ylabel("people", color="#1C2833")

        # visualization
        # TODO: should be better
        if max_scale:
            ax.set_ylim((0, total_size))
        ax.grid()

        for s in self.supervisables:
            s.plot(ax)
        ax.legend()

        # showing and saving the graph
        if save:
            fig.savefig(
                f"{output_dir}{total_size} agents, applying quarantine = {self.manager.consts.active_quarantine}, max scale = {max_scale}"
            )
        if auto_show:
            plt.show()


class Supervisable(ABC):
    @abstractmethod
    def snapshot(self, manager):
        pass

    @abstractmethod
    def scale(self) -> Optional[Tuple[float, float]]:
        pass

    @abstractmethod
    def plot(self, ax):
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        pass

    @classmethod
    @lru_cache
    def coerce(cls, arg, manager) -> Supervisable:
        if isinstance(arg, str):
            return _StateSupervisable(manager.medical_machine[arg])
        if isinstance(arg, cls.Delayed):
            inner: FloatSupervisable = cls.coerce(arg.arg, manager)
            return _DelayedSupervisable(inner, arg.delay)
        if isinstance(arg, cls.SumStates):
            supervisables: List[Supervisable] = []
            for state in arg.states:
                supervisables.append(cls.coerce(state, manager))
            return _SumStatesSupervisable(supervisables)
        raise TypeError

    class Delayed(NamedTuple):
        arg: Any
        delay: int

    class SumStates(NamedTuple):
        states: Sequence[str]


SupervisableMaker = Callable[[Any], Supervisable]


class FloatSupervisable(Supervisable):
    def __init__(self):
        self.x = []
        self.y = []

    @abstractmethod
    def get(self, manager) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def snapshot(self, manager):
        self.x.append(manager.current_date)
        self.y.append(self.get(manager))

    def scale(self):
        if not self.y:
            return None
        return min(self.y), max(self.y)

    def plot(self, ax):
        # todo preferred color/style?
        ax.plot(self.x, self.y, label=self.name())


class _StateSupervisable(FloatSupervisable):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def is_finished(self) -> bool:
        return isinstance(self.state, TerminalState) or (self.y and self.y[-1] == 0)

    def get(self, manager) -> float:
        return self.state.agent_count

    def name(self) -> str:
        return self.state.name


class _DelayedSupervisable(FloatSupervisable):
    def __init__(self, inner: FloatSupervisable, delay: int):
        super().__init__()
        self.inner = inner
        self.delay = delay

    def get(self, manager) -> float:
        desired_date = manager.current_date - self.delay
        desired_index = bisect(self.inner.x, desired_date)
        if desired_index >= len(self.inner.x):
            return np.nan
        return self.inner.y[desired_index]

    def name(self) -> str:
        return self.inner.name() + f" + {self.delay} days"

    def is_finished(self) -> bool:
        return True


class _SumStatesSupervisable(FloatSupervisable):
    def __init__(self, states_supervisables):
        super().__init__()
        self.states_supervisables = states_supervisables

    def is_finished(self) -> bool:
        return True

    def get(self, manager) -> float:
        return sum(s.get(manager) for s in self.states_supervisables)

    def name(self) -> str:
        ret_str = "sum of: "
        for s in self.states_supervisables:
            ret_str = ret_str + s.name() + ", "
        ret_str = ret_str[0:-2]
        return ret_str
