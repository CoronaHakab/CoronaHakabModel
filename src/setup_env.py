import subprocess
import sys
from warnings import warn

if sys.version_info < (3, 8, 0):
    raise Exception("must have python 3.8+")

requirements = [
    "numpy",
    "scipy",
    "scikit-learn"
]

extras = {
    "profiling": ["yappi"]
}

if not subprocess.run([sys.executable, "-m", "pip", "install", *requirements]):
    exit()

for k, v in extras.items():
    if not subprocess.run([sys.executable, "-m", "pip", "install", *v]):
        warn(f"installation of {k} extras failed, {k} will not be available")
