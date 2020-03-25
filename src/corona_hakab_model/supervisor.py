from typing import Dict, Iterable, List

import matplotlib_set_backend  # noqa: F401
from consts import Consts
from medical_state import MedicalState

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

    def __init__(self, states_to_track: Iterable[MedicalState]):
        self.state_history: Dict[MedicalState, List[int]] = {
            s: [] for s in states_to_track
        }
        self.xs = []
        self.max_height = -float("inf")

    def snapshot(self, manager):
        self.xs.append(manager.current_date)
        for s, arr in self.state_history.items():
            arr.append(s.agent_count)

    def plot(self, max_scale=True, auto_show=True, save=True):
        output_dir = "../output/"
        total_size = Consts.population_size
        title = f"Infections vs. Days, size={total_size}"
        if max_scale:
            height = total_size
        else:
            height = max(max(a) for a in self.state_history.values())
        text_height = height / 2

        # policies
        if Consts.active_quarantine:
            title = title + "\napplying lockdown from day {} to day {}".format(
                Consts.stop_work_days, Consts.resume_work_days
            )
            plt.axvline(x=Consts.stop_work_days, color="#0000ff")
            plt.text(
                Consts.stop_work_days + 2,
                text_height,
                f"day {Consts.stop_work_days} - pause all work",
                rotation=90,
            )
            plt.axvline(x=Consts.resume_work_days, color="#0000cc")
            plt.text(
                Consts.resume_work_days + 2,
                text_height,
                f"day {Consts.resume_work_days} - resume all work",
                rotation=90,
            )
        if Consts.home_quarantine_sicks:
            title = (
                title
                + "\napplying home quarantine for confirmed cases ({} of cases)".format(
                    Consts.caught_sicks_ratio
                )
            )
        if Consts.full_quarantine_sicks:
            title = (
                title
                + "\napplying full quarantine for confirmed cases ({} of cases)".format(
                    Consts.caught_sicks_ratio
                )
            )

        # plot parameters
        plt.title(title)
        plt.xlabel("days", color="#1C2833")
        plt.ylabel("people", color="#1C2833")

        # visualization
        # TODO: should be better
        if max_scale:
            plt.ylim((0, total_size))
        plt.grid()

        for (state, arr) in self.state_history.items():
            plt.plot(self.xs, arr, label=state.name)
        plt.legend()

        # showing and saving the graph
        if save:
            plt.savefig(
                f"{output_dir}{total_size} agents, applying quarantine = {Consts.active_quarantine}, max scale = {max_scale}"
            )
        if auto_show:
            plt.show()
