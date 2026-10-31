# Run an author script with fixed CPU settings.

import os
import random
import runpy
import sys
from pathlib import Path

for variable in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[variable] = "2"
random.seed(20261029)
entry = Path(sys.argv[1]).resolve()
sys.argv = [str(entry)] + sys.argv[2:]
sys.path.insert(0, str(entry.parent))
runpy.run_path(str(entry), run_name="__main__")
