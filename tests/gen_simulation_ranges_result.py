import json
from copy import deepcopy

from numpy import inf

from consts import Consts
from manager import SimulationManager


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
            self._max = 1.1 * x + 10
        if x <= self._min:
            self._min = 1.1 * x - 10

    def to_str(self):
        return f"Range({self._min}, {self._max})"

    def to_dict(self):
        return {'min': self._min, 'max': self._max}


test_states = dict(
    Symptomatic=Range(inf, -inf),
    Deceased=Range(inf, -inf),
    Asymptomatic=Range(inf, -inf),
    Hospitalized=Range(inf, -inf),
    ICU=Range(inf, -inf),
    Latent=Range(inf, -inf),
    Silent=Range(inf, -inf),
    Recovered=Range(inf, -inf)
)


def test_no_policy_simulation():
    consts = Consts(active_isolation=False)
    range_per_day_dict = [deepcopy(test_states) for _ in range(consts.total_steps)]
    for i in range(1_000):
        print(i)
        sm = SimulationManager(supervisable_makers=test_states.keys(), consts=consts)
        sm.run()

        for supervisable in sm.supervisor.supervisables:
            name = supervisable.name()
            val_per_day = supervisable.y
            for test_state, val in zip(range_per_day_dict, val_per_day):
                test_state[name].update(val)

    for i in range(consts.total_steps):
        range_per_day_dict[i] = {state: val.to_dict() for state, val in range_per_day_dict[i].items()}

    with open('simulation_test_ranges.json', 'w') as f:
        json.dump(range_per_day_dict, f)
