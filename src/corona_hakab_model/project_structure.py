from pathlib import Path

MODEL_FOLDER = Path(__file__).parent.parent
print('MODEL_FOLDER: {}'.format(MODEL_FOLDER))
OUTPUT_FOLDER = MODEL_FOLDER.parent / "output"
SIM_OUTPUT_FOLDER = OUTPUT_FOLDER / "sim_records"
