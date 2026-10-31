# Generate Shapes data and run the author's training and evaluation.

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from prepare import ROOT, prepare, sha256


def run_step(label, entry, args, source, out):
    command = [
        sys.executable,
        str(ROOT / "run_author.py"),
        str(source / entry),
        *map(str, args),
    ]
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "2"
    env["PYTHONUTF8"] = "1"
    start = time.time()
    print(f"Running {label}...", flush=True)
    with (out / f"{label}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    text = (out / f"{label}.log").read_text(encoding="utf-8")
    step = {
        "stage": label,
        "entry": entry,
        "arguments": list(map(str, args)),
        "exit_code": result.returncode,
        "elapsed_seconds": round(time.time() - start, 3),
        "log": f"{label}.log",
    }
    if result.returncode:
        print(text[-10000:])
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}; see {out / (label + '.log')}"
        )
    return step, text


def data_summary(path):
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        lengths = [len(f[key]["action"]) for key in f]
        first = f["0"]
        assert first["obs"].shape[1:] == (3, 50, 50)
        actions = set()
        for key in f:
            episode = f[key]
            obs, next_obs = episode["obs"][:], episode["next_obs"][:]
            assert np.isfinite(obs).all() and np.isfinite(next_obs).all()
            assert np.array_equal(obs[1:], next_obs[:-1])
            actions.update(episode["action"][:].tolist())
        assert actions.issubset(set(range(20)))
        return {
            "episodes": len(f),
            "transitions": sum(lengths),
            "steps_per_episode": sorted(set(lengths)),
            "observation_shape": list(first["obs"].shape[1:]),
            "observed_action_ids": sorted(actions),
            "sha256": sha256(path),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-scale",
        action="store_true",
        help="Use the author README's Shapes dataset sizes and default 100 training epochs.",
    )
    args = parser.parse_args()
    source = prepare()
    mode = "paper_scale" if args.paper_scale else "quick"
    out = ROOT / "outputs" / mode
    out.mkdir(parents=True, exist_ok=True)
    train_episodes, eval_episodes, epochs, batch = (
        (1000, 10000, 100, 1024) if args.paper_scale else (20, 100, 3, 128)
    )
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    train_data, eval_data = data_dir / "shapes_train.h5", data_dir / "shapes_eval.h5"
    result = {
        "status": "running",
        "mode": mode,
        "author_model": "C-SWM",
        "source_commit": json.loads((ROOT / "source.json").read_text())["commit"],
        "data_source": "Original author's ShapesTrain-v0 and ShapesEval-v0 simulator, via original data_gen/env.py",
        "cpu_threads": 2,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "gym", "scikit-image", "h5py"]
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch,
            "hidden_dim": 512,
            "embedding_dim": 2,
            "num_objects": 5,
            "encoder": "small",
            "learning_rate": 0.0005,
            "seed": 42,
        },
        "steps": [],
    }
    result["source_manifest_sha256"] = sha256(ROOT / "source.json")
    result["pixi_lock_sha256"] = sha256(ROOT / "pixi.lock")
    result["training"][
        "initialization"
    ] = "Author random initialization; parameters trained from scratch on the native Shapes training data."
    started = time.time()
    try:
        for label, env_id, path, count, seed in [
            ("generate_train", "ShapesTrain-v0", train_data, train_episodes, 1),
            ("generate_eval", "ShapesEval-v0", eval_data, eval_episodes, 2),
        ]:
            step, _ = run_step(
                label,
                "data_gen/env.py",
                [
                    "--env_id",
                    env_id,
                    "--fname",
                    path,
                    "--num_episodes",
                    count,
                    "--seed",
                    seed,
                ],
                source,
                out,
            )
            result["steps"].append(step)
        result["datasets"] = {
            "train": data_summary(train_data),
            "evaluation": data_summary(eval_data),
        }
        assert result["datasets"]["train"]["steps_per_episode"] == [100]
        assert result["datasets"]["evaluation"]["steps_per_episode"] == [10]
        checkpoints = out / "checkpoints"
        step, text = run_step(
            "train",
            "train.py",
            [
                "--dataset",
                train_data,
                "--encoder",
                "small",
                "--name",
                "shapes",
                "--save-folder",
                checkpoints,
                "--epochs",
                epochs,
                "--batch-size",
                batch,
                "--no-cuda",
            ],
            source,
            out,
        )
        result["steps"].append(step)
        losses = [float(x) for x in re.findall(r"Average loss: ([0-9.eE+-]+)", text)]
        if len(losses) != epochs or not all(math.isfinite(x) for x in losses):
            raise RuntimeError("Expected finite losses for every training epoch")
        result["training"]["author_reported_average_losses"] = losses
        result["training"]["optimizer_steps"] = (
            math.ceil(train_episodes * 100 / batch) * epochs
        )
        result["checkpoint_sha256"] = sha256(checkpoints / "shapes" / "model.pt")
        result["evaluation"] = {}
        for horizon in (1, 5, 10):
            step, text = run_step(
                f"evaluate_{horizon}",
                "eval.py",
                [
                    "--dataset",
                    eval_data,
                    "--save-folder",
                    checkpoints / "shapes",
                    "--num-steps",
                    horizon,
                    "--no-cuda",
                ],
                source,
                out,
            )
            result["steps"].append(step)
            hits = float(re.search(r"Hits @ 1: ([0-9.eE+-]+)", text).group(1))
            mrr = float(re.search(r"MRR: ([0-9.eE+-]+)", text).group(1))
            evaluated = int(
                re.search(r"Size of current topk evaluation batch: (\d+)", text).group(
                    1
                )
            )
            assert evaluated == eval_episodes and 0 <= hits <= 1 and 0 <= mrr <= 1
            result["evaluation"][str(horizon)] = {
                "hits_at_1": hits,
                "mrr": mrr,
                "episodes_evaluated": evaluated,
            }
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = round(time.time() - started, 3)
        (out / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(out / "result.json"),
                "evaluation": result["evaluation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
