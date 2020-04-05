import json
import os

from consts import Consts
from manager import SimulationManager

folder = os.path.dirname(__file__)
f = open(os.path.join(folder, 'simulation_test_ranges.json'), 'r')
range_per_day_dict = json.load(f)


def test_no_policy_simulation():
    """
    run the simulation with no policy and fixed R0
    Ranges for each state was generate from real sim runs, might be flaky.
    """
    consts = Consts(active_isolation=False, population_size=1_000, r0=2.4)
    keys = range_per_day_dict[0].keys()
    sm = SimulationManager(supervisable_makers=keys, consts=consts)
    sm.run()

    for supervisable in sm.supervisor.supervisables:
        name = supervisable.name()
        val_per_day = supervisable.y
        # check for each day if value is in range
        for test_state, val in zip(range_per_day_dict, val_per_day):
            state = test_state[name]
            assert state['min'] <= val <= state['max']
