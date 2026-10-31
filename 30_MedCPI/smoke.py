# Run MedCPI on the MIMIC-IV demo data.

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
    result = {"status": "passed", "command": "pixi run --locked smoke"}
    for name in ["cohort", "mapping", "pipeline", "training"]:
        result[name] = json.loads(
            (ROOT / "verification" / f"{name}.json").read_text(encoding="utf-8")
        )
    result["data_source"] = json.loads(
        (dataset / "dataset.json").read_text(encoding="utf-8")
    )
    training = result["training"]
    assert (
        training["status"] == "completed"
        and training["reloaded_test_metrics_identical"]
    )
    assert result["mapping"]["native_outputs_replaced_by_wrapper"] == 0
    result["model"] = "MedCPI"
    result["optimizer_updates_per_training_run"] = (
        (training["num_train_examples"] + training["batch_size"] - 1)
        // training["batch_size"]
    ) * training["epochs_ran"]
    (ROOT / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-cache", type=Path)
    parser.add_argument("--public-cache", type=Path)
    parser.add_argument("--sapbert-cache", type=Path)
    args = parser.parse_args()
    dataset = ensure_demo(cache=args.demo_cache)
    env = dict(
        os.environ,
        KG_MODEL_DATASET="MIMIC-IV Clinical Database Demo 2.2 (public)",
        KG_MODEL_MIMIC_VERSION="2.2",
    )
    command = [
        sys.executable,
        str(ROOT / "reproduce.py"),
        "--mimic-root",
        str(dataset),
        "--patients",
        "100",
        "--epochs",
        "1",
    ]
    for key in ["public_cache", "sapbert_cache"]:
        if getattr(args, key):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    write_summary(dataset)
    print(
        "MedCPI smoke passed. Summary: smoke_result.json; model and patient graphs: .private/"
    )


if __name__ == "__main__":
    main()
