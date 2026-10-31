# Run the author's stages; patient files and logs stay in .private.

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def environment():
    env = dict(
        os.environ,
        MEDCPI_PROJECT_ROOT=str(ROOT / "author"),
        MEDCPI_OUTPUT_ROOT=str(ROOT / ".private/outputs"),
        MEDCPI_MIMIC4_DIR=str(ROOT / ".private/raw/mimic4"),
        MEDCPI_ASSETS_ROOT=str(ROOT / ".assets"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        PYTHONHASHSEED="0",
    )
    return env


def run(name, arguments):
    log = ROOT / ".private/logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "author/scripts" / arguments[0]),
                *arguments[1:],
            ],
            env=environment(),
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    result = {
        "stage": name,
        "exit_code": p.returncode,
        "seconds": round(time.time() - start, 3),
    }
    (ROOT / "verification").mkdir(exist_ok=True)
    (ROOT / "verification" / f"{name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)
    if p.returncode:
        raise RuntimeError(f"Author stage {name} failed; private log retained.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=["preprocess", "aligned", "construct", "personalize", "integrate"],
    )
    a = ap.parse_args()
    if a.stage == "preprocess":
        run("preprocess", ["preprocess_ehr_to_jsonl.py", "--dataset", "mimic4"])
        run("task_instances", ["build_task_instances.py", "--dataset", "mimic4"])
        run("task_splits", ["create_task_splits.py", "--dataset", "mimic4"])
        p = ROOT / ".private/outputs/splits/mimic4/mortality/splits.json"
        stats = json.loads(p.read_text(encoding="utf-8"))
        public = {
            k: stats[k]
            for k in [
                "dataset",
                "task",
                "seed",
                "train_ratio",
                "valid_ratio",
                "num_source_rows",
                "num_rows_after_task_filter",
                "split_stats",
            ]
        }
        (ROOT / "verification/task_labels.json").write_text(
            json.dumps(public, indent=2), encoding="utf-8"
        )
        print(json.dumps(public), flush=True)
    if a.stage == "aligned":
        run(
            "aligned",
            ["build_aligned_patient_ehr.py", "--dataset", "mimic4", "--overwrite"],
        )
    if a.stage == "construct":
        out = ROOT / ".private/outputs/construct/shared"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            ROOT / "author/outputs/construct/shared", out, dirs_exist_ok=True
        )
        run(
            "construct",
            [
                "run_construct_pipeline.py",
                "--dataset",
                "mimic4",
                "--tasks",
                "mortality",
                "--skip-aligned",
                "--skip-inventory",
            ],
        )
    if a.stage == "personalize":
        run(
            "personalize",
            [
                "build_patient_pkg.py",
                "--dataset",
                "mimic4",
                "--task",
                "mortality",
                "--split",
                "all",
                "--attach-bridge-paths",
            ],
        )
    if a.stage == "integrate":
        run(
            "prepare_integrate",
            [
                "prepare_integrate_inputs.py",
                "--dataset",
                "mimic4",
                "--task",
                "mortality",
                "--split",
                "all",
                "--build-manifest",
            ],
        )
        for split in ["train", "valid", "test"]:
            run(
                "integrate_data_" + split,
                [
                    "integrate_data.py",
                    "--dataset",
                    "mimic4",
                    "--task",
                    "mortality",
                    "--split",
                    split,
                    "--build-indexes",
                ],
            )


if __name__ == "__main__":
    main()
