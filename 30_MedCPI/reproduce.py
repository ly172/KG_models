# Prepare public resources, select a MIMIC cohort and run the MedCPI pipeline.

import argparse
import os
from pathlib import Path
import subprocess
import sys
from run_author_stages import environment

ROOT = Path(__file__).resolve().parent


def run(*args):
    subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, env=environment(), check=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--public-cache", type=Path)
    ap.add_argument("--sapbert-cache", type=Path)
    ap.add_argument("--patients", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    resources = ROOT / ".assets/public_resources"
    run("fetch_author.py")
    args = ["public_resources/prepare_public_resources.py", "--output", resources]
    if a.public_cache:
        args += ["--cache", a.public_cache]
    run(*args)
    args = ["fetch_sapbert.py"]
    if a.sapbert_cache:
        args += ["--cache", a.sapbert_cache]
    run(*args)
    run("prepare_cohort.py", "--mimic-root", a.mimic_root, "--patients", a.patients)
    run("run_author_stages.py", "preprocess")
    run(
        "prepare_public_inputs.py",
        "--mimic-root",
        a.mimic_root,
        "--resources-root",
        resources,
        "--graphcare-root",
        resources / "graphcare",
        "--semgroups",
        resources / "SemGroups.txt",
    )
    run(
        "run_original_mapper.py",
        "--sapbert-root",
        ROOT / ".assets/sapbert",
        "--source-manifest",
        "verification/sapbert_source.json",
    )
    for stage in ["aligned", "construct", "personalize", "integrate"]:
        run("run_author_stages.py", stage)
    run("train.py", "--epochs", a.epochs)
    run("verify_pipeline.py")


if __name__ == "__main__":
    main()
