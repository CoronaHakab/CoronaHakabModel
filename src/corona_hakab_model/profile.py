import subprocess
from pathlib import Path
from time import time
from warnings import warn
from project_structure import MODEL_FOLDER

import yappi
from manager import SimulationManager

qcachegrind_path = Path(r"D:\chrome downloads\qcachegrind074-32bit-x86\qcachegrind074-x86\qcachegrind.exe")
profile_gen = False

if __name__ == "__main__":
    if profile_gen:
        yappi.start()
    sm = SimulationManager(
        ("Recovered", "Deceased", "Symptomatic", "Asymptomatic", "Hospitalized", "ICU", "Latent", "Silent",)
    )
    if not profile_gen:
        yappi.start()
    sm.run()

    stats = yappi.get_func_stats()
    dest_path = MODEL_FOLDER / "profilings/callgrind.out." + str(int(time()))
    stats.save(dest_path, "CALLGRIND")

    if not qcachegrind_path:
        pass
    if qcachegrind_path.exists():
        subprocess.Popen([str(qcachegrind_path.absolute()), dest_path], close_fds=True)
    else:
        warn(f"set qcachegrind_path to a valid path to open results")
