from pathlib import Path

MODEL_FOLDER = Path(__file__).parent
OUTPUT_FOLDER = MODEL_FOLDER.parent.parent / "output"
SIM_OUTPUT_FOLDER = OUTPUT_FOLDER / "sim_records"
ANALYZERS_FOLDER = MODEL_FOLDER / "analyzers"
