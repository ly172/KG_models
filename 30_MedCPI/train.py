# Run the author's training, fix the report error and compare checkpoints.

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_author_stages import environment

ROOT = Path(__file__).resolve().parent


def source_hash_metadata(source, patched):
    return {
        prefix + "_" + kind + "_sha256": hashlib.sha256(value).hexdigest()
        for prefix, path in [("original", source), ("patched", patched)]
        for kind, value in [
            ("file", path.read_bytes()),
            ("normalized_text", path.read_text(encoding="utf-8").encode()),
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    source = ROOT / "author/scripts/train_integrate_formal.py"
    original = source.read_text(encoding="utf-8")
    removals = [
        '        "num_parameters": seed_results[0]["num_parameters"],\n',
        '        "num_trainable_parameters": seed_results[0]["num_trainable_parameters"],\n',
    ]
    fixed = original
    for line in removals:
        if fixed.count(line) != 1:
            raise RuntimeError(
                "Pinned author report lines differ; inspect before patching"
            )
        fixed = fixed.replace(line, "")
    # The two removed entries duplicate fields already in the report.
    runtime = ROOT / ".private/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    patched = runtime / "train_integrate_formal.py"
    patched.write_text(fixed, encoding="utf-8")
    (ROOT / "verification/trainer_report_only.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                fixed.splitlines(True),
                fromfile="author/train_integrate_formal.py",
                tofile="runtime/train_integrate_formal.py",
            )
        ),
        encoding="utf-8",
    )
    env = environment()
    env["PYTHONPATH"] = str(ROOT / "author/scripts")
    attempts = []
    for label, path in [("original", source), ("report_fixed", patched)]:
        log = ROOT / ".private/logs" / f"train_{label}.log"
        start = time.time()
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.run(
                [
                    sys.executable,
                    str(path),
                    "--dataset",
                    "mimic4",
                    "--task",
                    "mortality",
                    "--epochs",
                    str(a.epochs),
                    "--seed",
                    "0",
                    "--output-tag",
                    label,
                ],
                env=env,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        raw = log.read_text(encoding="utf-8")
        expected_failure = (
            "NameError" in raw and "name 'seed_results' is not defined" in raw
        )
        item = {
            "version": label,
            "exit_code": p.returncode,
            "seconds": round(time.time() - start, 3),
            "expected_report_NameError": expected_failure,
        }
        attempts.append(item)
        print(json.dumps(item), flush=True)
        (ROOT / "verification/training_attempts.json").write_text(
            json.dumps(attempts, indent=2), encoding="utf-8"
        )
        if label == "original" and not (p.returncode and expected_failure):
            raise RuntimeError(
                "Original run differs from expected report failure; inspect private log"
            )
        if label == "report_fixed" and p.returncode:
            raise RuntimeError("Fixed run failed; inspect private log")
    os.environ.update(env)
    sys.path.insert(0, str(ROOT / "author/scripts"))
    import torch
    import train_integrate_formal as native
    from integrate_model import MedCPIIntegrateModel, task_granularity

    task = ROOT / ".private/outputs/integrate/mimic4/mortality"
    saved = {
        label: torch.load(
            task / f"best_valid_formal_{label}_seed0.pt",
            map_location="cpu",
            weights_only=False,
        )
        for label in ["original", "report_fixed"]
    }
    same = all(
        torch.equal(value, saved["report_fixed"]["model_state_dict"][key])
        for key, value in saved["original"]["model_state_dict"].items()
    )
    if not same:
        raise AssertionError("Original and repaired model checkpoints differ")
    checkpoint = saved["report_fixed"]
    relation_vocab = native.build_relation_vocab(
        native.CONSTRUCT_ROOT / "shared/mortality/relation_schema.json"
    )
    dataset = native.LazyIntegrateDataset(
        "mimic4", "mortality", "test", checkpoint["concept_vocab"], relation_vocab
    )
    model = MedCPIIntegrateModel(
        concept_vocab_size=len(checkpoint["concept_vocab"]["concept_to_id"]),
        relation_vocab_size=checkpoint["relation_vocab_size"],
        **{
            k: checkpoint[k]
            for k in [
                "embedding_dim",
                "hidden_dim",
                "num_heads",
                "num_htt_layers",
                "num_rgcn_layers",
                "dropout",
            ]
        },
        granularity=task_granularity("mortality"),
        fusion_strategy="cross_attention",
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = native.collect_eval_outputs(
        model,
        native.build_loader_for_seed(dataset, 32, False, 0),
        torch.nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
    )
    progress = json.loads(
        (task / "formal_train_progress_report_fixed_seed0.json").read_text(
            encoding="utf-8"
        )
    )
    same_metrics = all(
        abs(metrics[k] - v) < 1e-10 if isinstance(v, (int, float)) else metrics[k] == v
        for k, v in progress["test_metrics"].items()
    )
    if not same_metrics:
        raise AssertionError("Reloaded test metrics differ")
    public = {
        k: v
        for k, v in progress.items()
        if k not in ["best_checkpoint_path", "output_suffix"]
    }
    public.update(
        original_and_fixed_state_tensors_identical=same,
        reloaded_test_metrics_identical=same_metrics,
        repair="Removed two duplicate report dictionary fields referencing undefined seed_results. Existing parameter counts remain; model, optimizer, loss, batches and selection unchanged.",
    )
    public.update(source_hash_metadata(source, patched))
    (ROOT / "verification/training.json").write_text(
        json.dumps(public, indent=2), encoding="utf-8"
    )
    print(json.dumps(public), flush=True)


if __name__ == "__main__":
    main()
