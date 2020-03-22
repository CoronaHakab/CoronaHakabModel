import subprocess
import sys

if sys.version_info < (3,8,0):
    raise Exception("must have python 3.8+")

requirements = [
    "numpy",
    "scipy",
    "scikit-learn",
    "yappi"
]

subprocess.run([sys.executable, "-m", "pip", "install", *requirements])