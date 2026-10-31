# Train and evaluate GAMENet on demo or generated visits.

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from prepare_author import ROOT, prepare, sha256
from simulate_inputs import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patients", type=int, default=36)
    parser.add_argument("--public-demo", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patients < 12 or args.patients % 6:
        parser.error("Use positive epochs and a patient count >= 12 divisible by 6")
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        KG_DEVICE="cpu",
        KG_EPOCHS=str(args.epochs),
    )
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[name] = "2"
    author = prepare()
    if args.public_demo:
        from public_demo import generate as public_inputs

        summary = public_inputs(author)
    else:
        summary = generate(author, args.patients)
    output = ROOT / ("outputs_demo" if args.public_demo else "outputs")
    output.mkdir(exist_ok=True)
    result = {
        "status": "running",
        "model": "Author GAMENet with DDI",
        "inputs": summary,
        "epochs": args.epochs,
        "embedding_dim": 64,
        "device": "cpu",
        "steps": [],
    }
    commands = [
        (
            "train",
            [sys.executable, "train_GAMENet.py", "--model_name", "GAMENet", "--ddi"],
        ),
        (
            "test",
            [
                sys.executable,
                "train_GAMENet.py",
                "--model_name",
                "GAMENet",
                "--ddi",
                "--eval",
                "--resume_path",
                "saved/GAMENet/final.model",
            ],
        ),
    ]
    try:
        for name, command in commands:
            print("Running author " + name, flush=True)
            start = time.time()
            with (output / (name + ".log")).open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command,
                    cwd=author / "code",
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            result["steps"].append(
                {
                    "stage": name,
                    "exit_code": process.returncode,
                    "seconds": round(time.time() - start, 3),
                }
            )
            if process.returncode:
                raise RuntimeError(
                    "Author " + name + " failed; see " + str(output / (name + ".log"))
                )
        text = (output / "test.log").read_text(encoding="utf-8")
        match = re.search(
            r"DDI Rate: ([0-9.eE+-]+), Jaccard: ([0-9.eE+-]+),\s+PRAUC: ([0-9.eE+-]+)",
            text,
        )
        if not match or not all(math.isfinite(float(x)) for x in match.groups()):
            raise RuntimeError("Expected finite author evaluation output")
        result["metrics"] = dict(
            zip(["ddi_rate", "jaccard", "prauc"], map(float, match.groups()))
        )
        result["checkpoint_sha256"] = sha256(author / "code/saved/GAMENet/final.model")
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
