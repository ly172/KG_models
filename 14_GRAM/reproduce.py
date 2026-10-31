# Run GRAM on a small MIMIC-IV cohort.

import argparse
import json
import math
import os
import pickle
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

from adapt_mimic import adapt
from prepare_author import ROOT, prepare, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", default=os.environ.get("MIMIC_IV_ROOT"))
    parser.add_argument("--patients", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--run-label", default="current")
    args = parser.parse_args()
    if not args.mimic_root:
        parser.error(
            "Provide --mimic-root or set MIMIC_IV_ROOT. Use the smoke task to download the public MIMIC-IV Demo."
        )
    if not re.fullmatch("[A-Za-z0-9_]+", args.run_label):
        parser.error("Run label must use letters, digits, or underscores")
    author = prepare()
    private = ROOT / ".private" / args.run_label
    private.mkdir(parents=True, exist_ok=True)
    admissions, diagnoses, cohort = adapt(
        args.mimic_root, private / "inputs", args.patients
    )
    ccs = ROOT / "resources/ccs_multi_dx_tool_2015.csv"
    resources = json.loads(
        (ROOT / "resources/source.json").read_text(encoding="utf-8-sig")
    )
    assert sha256(ccs) == resources["sha256"]
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="0",
    )
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[name] = "2"
    cache = ROOT / ".private/theano_cache"
    env["THEANO_FLAGS"] = (
        "blas.ldflags=,cxx=,mode=FAST_COMPILE,linker=py,floatX=float64,compiledir="
        + str(cache)
    )
    result = {
        "status": "running",
        "model": "Author GRAM",
        "input_kind": "real_MIMIC_IV_hospital_data",
        "cohort": cohort,
        "model_settings": {
            "embedding": 128,
            "gru": 128,
            "attention": 128,
            "batch_size": 100,
            "dropout_rate": 0.5,
            "L2": 0.001,
            "epochs": args.epochs,
            "floatX": "float64",
        },
        "steps": [],
    }

    def execute(name, file, arguments):
        started = time.time()
        command = [
            sys.executable,
            str(ROOT / "run_author.py"),
            str(author / file),
        ] + list(map(str, arguments))
        print("Running " + name + "...", flush=True)
        with (private / (name + ".log")).open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command, cwd=author, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        result["steps"].append(
            {
                "stage": name,
                "author_entry": file,
                "exit_code": process.returncode,
                "seconds": round(time.time() - started, 3),
            }
        )
        if process.returncode:
            raise RuntimeError(name + " failed; inspect the local private log")

    try:
        prefix = private / "inputs/mimic"
        tree = private / "inputs/tree"
        execute("process_mimic", "process_mimic.py", [admissions, diagnoses, prefix])
        execute(
            "build_trees",
            "build_trees.py",
            [ccs, str(prefix) + ".seqs", str(prefix) + ".types", tree],
        )
        with open(str(tree) + ".seqs", "rb") as stream:
            sequences = pickle.load(stream)
        count = len(sequences)
        train_count = count - int(count * 0.2) - int(count * 0.1)
        result["author_split"] = {
            "train_patients": train_count,
            "validation_patients": int(count * 0.1),
            "test_patients": int(count * 0.2),
            "optimizer_steps_per_epoch": math.ceil(train_count / 100),
            "next_visit_prediction_pairs": sum(
                len(patient) - 1 for patient in sequences
            ),
        }
        execute(
            "train",
            "gram.py",
            [
                str(tree) + ".seqs",
                str(prefix) + ".3digitICD9.seqs",
                tree,
                private / "model",
                "--n_epochs",
                args.epochs,
                "--verbose",
            ],
        )
        text = (private / "train.log").read_text(encoding="utf-8")
        values = re.findall(
            r"^Epoch:\d+, Duration:[0-9.eE+-]+, Train_Cost:([0-9.eE+-]+), Valid_Cost:([0-9.eE+-]+), Test_Cost:([0-9.eE+-]+)",
            text,
            flags=re.MULTILINE,
        )
        if not values or not all(
            math.isfinite(float(value)) for triple in values for value in triple
        ):
            raise RuntimeError("Expected finite author training/validation/test costs")
        result["costs"] = [
            {"train": float(a), "validation": float(b), "test": float(c)}
            for a, b, c in values
        ]
        checkpoints = sorted(private.glob("model.*.npz"))
        if not checkpoints:
            raise RuntimeError("Author training did not save a model checkpoint")
        result["checkpoints"] = []
        for checkpoint in checkpoints:
            with np.load(checkpoint) as arrays:
                if not arrays.files or not all(
                    np.isfinite(arrays[key]).all() for key in arrays.files
                ):
                    raise RuntimeError(
                        "Saved model contains missing or non-finite parameter arrays"
                    )
                result["checkpoints"].append(
                    dict(
                        file=checkpoint.name,
                        sha256=sha256(checkpoint),
                        parameter_arrays=len(arrays.files),
                    )
                )
        result["random_settings"] = {
            "author_numpy_seed": 0,
            "author_dropout_stream_seed": 123,
            "wrapper_python_random_seed": 20261029,
            "wrapper_python_hash_seed": 0,
        }
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        (private / "aggregate_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
