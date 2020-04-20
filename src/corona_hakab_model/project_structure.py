from pathlib import Path
from datetime import datetime

MODEL_FOLDER = Path(__file__).parent.parent
print('MODEL_FOLDER: {}'.format(MODEL_FOLDER))
OUTPUT_FOLDER = MODEL_FOLDER.parent / "output" / (datetime.now().strftime("%Y%m%d-%H%M%S"))
SIM_OUTPUT_FOLDER = OUTPUT_FOLDER / "sim_records"
