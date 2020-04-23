from pathlib import Path
from datetime import datetime
import uuid
# todo: Yes, thus is fugly and should be changed
INTERACTIVE_MODE = False

prefix_output_dir_name = input("Enter your prefix without spaces (default: None): ") if INTERACTIVE_MODE else ''
main_dir_name = input("Enter the main dirname without spaces (default: datetime, for empty enter x): ") \
    if INTERACTIVE_MODE else ''
if main_dir_name == '':
    uuid_str = uuid.uuid4().hex[:8]  # hotfix in case we run consecutive runs
    main_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S")+"_"+uuid_str
elif main_dir_name.lower() == 'x':
    main_dir_name = ''
suffix_output_dir_name = input("Enter your suffix without spaces (default: None): ") if INTERACTIVE_MODE else ''
output_subdir = prefix_output_dir_name + main_dir_name + suffix_output_dir_name

SOURCE_FOLDER = Path(__file__).parent.parent
MODEL_FOLDER = SOURCE_FOLDER.parent
OUTPUT_FOLDER = MODEL_FOLDER / "output" / output_subdir
SIM_OUTPUT_FOLDER = OUTPUT_FOLDER / "sim_records"

print('MODEL FOLDER: {}'.format(MODEL_FOLDER))
print('SOURCE FOLDER: {}'.format(SOURCE_FOLDER))
print('OUTPUT FOLDER: {}'.format(OUTPUT_FOLDER))
