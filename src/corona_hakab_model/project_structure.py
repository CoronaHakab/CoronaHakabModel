from pathlib import Path
from datetime import datetime

# todo: Yes, thus is fugly and should be changed
INTERACTIVE_MODE = False

prefix_output_dir_name = input("Enter your prefix without spaces (default: None): ") if INTERACTIVE_MODE else ''
main_dir_name = input("Enter the main dirname without spaces (default: datetime, for empty enter x): ") if INTERACTIVE_MODE else ''
if main_dir_name == '':
    main_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S")
elif main_dir_name.lower() == 'x':
    main_dir_name = ''
suffix_output_dir_name = input("Enter your suffix without spaces (default: None): ") if INTERACTIVE_MODE else ''
output_subdir = prefix_output_dir_name + main_dir_name + suffix_output_dir_name

MODEL_FOLDER = Path(__file__).parent
SOURCE_FOLDER = MODEL_FOLDER.parent
OUTPUT_FOLDER = MODEL_FOLDER.parent.parent / "output" / output_subdir
SIM_OUTPUT_FOLDER = OUTPUT_FOLDER / "sim_records"
ANALYZERS_FOLDER = MODEL_FOLDER / "analyzers"

print('MODEL FOLDER: {}'.format(MODEL_FOLDER))
print('SOURCE FOLDER: {}'.format(SOURCE_FOLDER))
print('OUTPUT FOLDER: {}'.format(OUTPUT_FOLDER))
