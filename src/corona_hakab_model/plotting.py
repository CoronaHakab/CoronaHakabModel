import matplotlib_set_backend  # noqa: F401
import numpy as np
from consts import Consts
from matplotlib import pyplot as plt
from medical_state import MedicalState
from sklearn.linear_model import LinearRegression


class StatisticsPlotter:
    """
    Plots statistics about the simulation.
    """

    def plot_infected_per_generation(self, per_generation, max_scale=True):
        output_dir = "../output/"
        total_size = Consts.population_size
        title = f"Infections vs. Days, size={total_size}"
        text_height = per_generation["total infected"][-1] // 2
        if max_scale:
            text_height = total_size / 2

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

        # data
        p1 = plt.plot(per_generation["total infected"])
        p2 = plt.plot(per_generation[MedicalState.Immune])
        p3 = plt.plot(per_generation[MedicalState.Deceased])
        p4 = plt.plot(per_generation["total current sick"])
        p5 = plt.plot(per_generation[MedicalState.Symptomatic])
        p6 = plt.plot(per_generation[MedicalState.Asymptomatic])
        p7 = plt.plot(per_generation[MedicalState.Hospitalized])
        p8 = plt.plot(per_generation[MedicalState.Icu])
        plt.legend(
            (p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0]),
            (
                "infected",
                "recovered",
                "dead",
                "currently sick",
                "symptomatic",
                "asymptomatic",
                "Hospitalized",
                "Icu",
            ),
        )

        # showing and saving the graph
        plt.savefig(
            f"{output_dir}{total_size} agents, applying quarantine = {Consts.active_quarantine}, max scale = {max_scale}"
        )
        plt.show()
        if max_scale:
            self.plot_infected_per_generation(per_generation, False)

    def plot_log_with_linear_regression(
        self,
        infected_per_generation_vector,
        recovered_per_generation,
        dead_per_generation,
    ):
        # fit only to the start of the model.
        # We will always start with exponent, then die down one way or another.
        diff_vector = np.diff(infected_per_generation_vector)
        x_start_search = np.argmax(diff_vector > 0)
        x_max = np.argmax(
            diff_vector[x_start_search:] <= 0
        )  # fit only until curvature starts going down.

        # get log vector
        infected_log_vector = np.log(infected_per_generation_vector)
        recovered_log_vector = np.log(recovered_per_generation)
        dead_log_vector = np.log(dead_per_generation)

        # put into variables for the fitting
        orig_x = np.array(range(infected_log_vector.size))
        x = orig_x[:x_max].reshape((-1, 1))  # used for the regression
        y = infected_log_vector[:x_max]

        # fit to linear
        model = LinearRegression()
        model.fit(x, y)

        # get fitting constants, y = b0 + b1*X
        b_0 = model.intercept_
        b_1 = model.coef_[0]
        r_sqr = model.score(x, y)  # R^2

        # plot the regression line
        regression_x = np.linspace(0, x_max, 100)
        regression_y = b_1 * regression_x + b_0
        plt.plot(
            regression_x,
            regression_y,
            "-r",
            label="{:.2f} + {:.2f}*x; R^2 = {:.2f}".format(b_0, b_1, r_sqr),
        )

        # plot the data itself
        p1 = plt.plot(orig_x, infected_log_vector)
        p2 = plt.plot(orig_x, recovered_log_vector)
        p3 = plt.plot(orig_x, dead_log_vector)

        plt.legend(
            (p1[0], p2[0], p3[0]),
            ("infected_log_vector", "recovered_log_vector", "dead_log_vector"),
        )

        # plot parameters
        plt.title("Log graph with regression")
        plt.xlabel("steps", color="#1C2833")
        plt.ylabel("log(people)", color="#1C2833")
        plt.legend(loc="upper left")

        plt.grid()

        plt.show()


class StatePlotter:
    """
    Plots a current state of the simulation.
    """

    pass
