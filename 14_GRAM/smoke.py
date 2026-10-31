# Run GRAM on the MIMIC-IV demo data.

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(
    0,
    str(
        next(
            parent / "shared"
            for parent in ROOT.parents
            if (parent / "shared/downloads.py").is_file()
        )
    ),
)
from mimic_demo import ensure_demo


def write_summary(dataset):
    result = json.loads(
        (ROOT / ".private/smoke/aggregate_result.json").read_text(encoding="utf-8")
    )
    result["command"] = "pixi run --locked smoke"
    result["data_source"] = json.loads(
        (dataset / "dataset.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "passed"
    result["optimizer_updates"] = (
        result["author_split"]["optimizer_steps_per_epoch"]
        * result["model_settings"]["epochs"]
    )
    (ROOT / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-cache", type=Path)
    args = parser.parse_args()
    dataset = ensure_demo(cache=args.demo_cache)
    env = dict(
        os.environ,
        KG_MODEL_DATASET="MIMIC-IV Clinical Database Demo 2.2 (public)",
        KG_MODEL_MIMIC_VERSION="2.2",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "reproduce.py"),
            "--mimic-root",
            str(dataset),
            "--patients",
            "100",
            "--epochs",
            "1",
            "--run-label",
            "smoke",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    write_summary(dataset)
    print(
        "GRAM smoke passed. Summary: smoke_result.json; model and logs: .private/smoke/"
    )


if __name__ == "__main__":
    main()
