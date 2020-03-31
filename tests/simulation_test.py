import json

from consts import Consts
from manager import SimulationManager

f = open('simulation_test_ranges.json', 'r')
range_per_day_dict = json.load(f)


def test_no_policy_simulation():
    consts = Consts(active_isolation=False)
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
