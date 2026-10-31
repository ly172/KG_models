# Run an author script with a fixed number of CPU threads.

import os
import runpy
import sys
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "2"
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
entry = Path(sys.argv[1]).resolve()
sys.argv = [str(entry), *sys.argv[2:]]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(entry.parent))
runpy.run_path(str(entry), run_name="__main__")
