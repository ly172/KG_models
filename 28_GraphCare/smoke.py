# Run GraphCare on the MIMIC-IV demo data.

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
        (ROOT / "real_verification/result.json").read_text(encoding="utf-8")
    )
    result["command"] = subprocess.list2cmdline(
        ["pixi", "run", "--locked", "smoke", *sys.argv[1:]]
    )
    result["data_source"] = json.loads(
        (dataset / "dataset.json").read_text(encoding="utf-8")
    )
    (ROOT / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-cache", type=Path)
    parser.add_argument("--asset-cache", type=Path)
    parser.add_argument("--encoded-cache", type=Path)
    parser.add_argument(
        "--precision",
        "--encoder-dtype",
        dest="encoder_dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    args = parser.parse_args()
    dataset = ensure_demo(cache=args.demo_cache)
    env = dict(
        os.environ,
        KG_MODEL_DATASET="MIMIC-IV Clinical Database Demo 2.2 (public)",
        KG_MODEL_MIMIC_VERSION="2.2",
    )
    command = [
        sys.executable,
        str(ROOT / "reproduce_real.py"),
        "--mimic-root",
        str(dataset),
        "--patients",
        "100",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--encoder-dtype",
        args.encoder_dtype,
        "--text-batch-size",
        "1" if args.encoder_dtype == "bfloat16" else "4",
    ]
    for key in ["asset_cache", "encoded_cache"]:
        if getattr(args, key):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    write_summary(dataset)
    print(
        "GraphCare smoke passed. Summary: smoke_result.json; model and vectors: .private/"
    )


if __name__ == "__main__":
    main()
