import json
from copy import deepcopy

from consts import Consts
from manager import SimulationManager
from numpy import inf
from tests.simulation_sanity_test.simulation_test import consts_file


class Range:
    """
    Data structure to hold min and max allowed values.
    Because our simulation contains random aspects we evaluate the tests
    using a continuous range of possible answers.
    """

    def __init__(self, _min: float, _max: float) -> None:
        self._min = _min
        self._max = _max

    def contains(self, x: float) -> bool:
        return self._min <= x <= self._max

    def update(self, x):
        if x >= self._max:
            self._max = x
        if x <= self._min:
            self._min = x

    def to_dict(self):
        return {"min": self._min * 0.9 - 10, "max": self._max * 1.1 + 10}  # take 10% margin


# initialize state to test and set ranges
test_states = dict(
    Symptomatic=Range(inf, -inf),
    Deceased=Range(inf, -inf),
    Asymptomatic=Range(inf, -inf),
    Hospitalized=Range(inf, -inf),
    ICU=Range(inf, -inf),
    Latent=Range(inf, -inf),
    Silent=Range(inf, -inf),
    Recovered=Range(inf, -inf),
)


def gen_no_policy_simulation():
    """
    for 500 rounds, run the simulation and update each day ranges according to the global min and max values per state.
    """
    consts = Consts.from_file(consts_file)
    range_per_day_dict = [deepcopy(test_states) for _ in range(consts.total_steps)]

    for i in range(500):
        print(i)
        sm = SimulationManager(supervisable_makers=test_states.keys(), consts=consts)
        sm.run()

        for supervisable in sm.supervisor.supervisables:
            name = supervisable.name()
            val_per_day = supervisable.y
            for test_state, val in zip(range_per_day_dict, val_per_day):
                test_state[name].update(val)

    # translate to dict
    for i in range(consts.total_steps):
        range_per_day_dict[i] = {state: val.to_dict() for state, val in range_per_day_dict[i].items()}
    # dump to json
    with open("simulation_test_ranges.json", "w") as f:
        json.dump(range_per_day_dict, f)


if __name__ == "__main__":
    gen_no_policy_simulation()
