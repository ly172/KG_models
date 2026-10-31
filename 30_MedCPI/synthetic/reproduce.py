# Train MedCPI on generated visits with the author's training function.

import argparse
import ast
import hashlib
import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import random
import sys

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "2"
from fetch_author import fetch_author

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--input-records", type=Path)
    parser.add_argument("--relation-count", type=int, default=6)
    parser.add_argument("--output-name", default="outputs")
    args = parser.parse_args()
    author = fetch_author()
    sys.path.insert(0, str(author / "scripts"))
    import torch
    import numpy as np
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_auc_score, average_precision_score
    from integrate_dataset import (
        IntegratePreviewDataset,
        build_concept_vocab,
        collate_integrate_batch,
    )
    from integrate_model import MedCPIIntegrateModel, task_granularity

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20261029)
    random.seed(20261029)
    out = ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)

    def sample(patient):
        visits = 2 + patient % 3
        ids = [
            "SYNTHETIC_CONCEPT_%02d" % ((patient + offset) % 32) for offset in range(9)
        ]
        nodes = [{"node_kind": "patient"}] + [
            {"node_kind": "visit"} for _ in range(visits)
        ]
        nodes += [{"node_kind": "concept", "concept_id": c} for c in ids]
        edges, sequence = [], []

        def edge(a, b, rel):
            edges.append(dict(src_index=a, dst_index=b, relation_id=rel))

        for visit in range(visits):
            selected = [(3 * visit + j) % len(ids) for j in range(3)]
            concepts = [
                dict(concept_id=ids[k], source_type=kind)
                for k, kind in zip(selected, ["diagnosis", "procedure", "medication"])
            ]
            day = datetime(2000, 1, 1) + timedelta(days=patient * 3 + visit * 60)
            sequence.append(
                dict(admittime=day.strftime("%Y-%m-%d %H:%M:%S"), concepts=concepts)
            )
            edge(0, 1 + visit, 0)
            edge(1 + visit, 0, 1)
            for k in selected:
                edge(1 + visit, 1 + visits + k, 2)
                edge(1 + visits + k, 1 + visit, 3)
        for k in range(len(ids) - 1):
            edge(1 + visits + k, 2 + visits + k, 4)
            edge(2 + visits + k, 1 + visits + k, 5)
        return dict(
            instance_id="SYNTHETIC_%03d" % patient,
            label=patient % 2,
            ehr_sequence=sequence,
            pkg_nodes=nodes,
            pkg_edges=edges,
        )

    records = (
        json.loads(args.input_records.read_text(encoding="utf-8"))
        if args.input_records
        else [sample(i) for i in range(48)]
    )
    assert len(records) == 48
    vocab = build_concept_vocab(records[:32])
    (out / "synthetic_samples.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    datasets = [
        IntegratePreviewDataset(s, vocab)
        for s in [records[:32], records[32:40], records[40:]]
    ]
    loaders = [
        DataLoader(
            ds, batch_size=8, shuffle=(i == 0), collate_fn=collate_integrate_batch
        )
        for i, ds in enumerate(datasets)
    ]
    config = dict(
        concept_vocab_size=len(vocab["concept_to_id"]),
        relation_vocab_size=args.relation_count,
        embedding_dim=64,
        hidden_dim=64,
        num_heads=4,
        num_htt_layers=1,
        num_rgcn_layers=2,
        dropout=0.1,
        granularity="visit",
        fusion_strategy="cross_attention",
    )
    original_source = (author / "scripts/train_integrate_formal.py").read_text(
        encoding="utf-8"
    )
    function_names = {
        "write_json",
        "artifact_name",
        "combine_suffix",
        "count_parameters",
        "resolve_selection_metric_value",
        "set_global_seed",
        "write_progress",
        "move_batch_to_device",
        "build_loader_for_seed",
        "collect_eval_outputs",
        "run_single_seed",
    }
    functions = [
        copy.deepcopy(n)
        for n in ast.parse(original_source).body
        if isinstance(n, ast.FunctionDef) and n.name in function_names
    ]
    run_function = next(n for n in functions if n.name == "run_single_seed")
    removed_report_fields = 0
    for node in ast.walk(run_function):
        if isinstance(node, ast.Dict):
            keep = [
                i
                for i, value in enumerate(node.values)
                if not any(
                    isinstance(n, ast.Name) and n.id == "seed_results"
                    for n in ast.walk(value)
                )
            ]
            removed_report_fields += len(node.keys) - len(keep)
            node.keys = [node.keys[i] for i in keep]
            node.values = [node.values[i] for i in keep]
    assert removed_report_fields == 2
    captured = {}

    def build_model(**options):
        model = MedCPIIntegrateModel(**options)
        captured["model"] = model
        captured["before"] = {
            k: v.detach().clone() for k, v in model.named_parameters()
        }
        return model

    scope = dict(
        argparse=argparse,
        json=json,
        random=random,
        Path=Path,
        np=np,
        torch=torch,
        nn=nn,
        AdamW=AdamW,
        DataLoader=DataLoader,
        Dataset=Dataset,
        roc_auc_score=roc_auc_score,
        average_precision_score=average_precision_score,
        collate_integrate_batch=collate_integrate_batch,
        task_granularity=task_granularity,
        MedCPIIntegrateModel=build_model,
    )
    module = ast.Module(body=functions, type_ignores=[])
    exec(
        compile(
            ast.fix_missing_locations(module),
            str(author / "scripts/train_integrate_formal.py"),
            "exec",
        ),
        scope,
    )
    train_args = argparse.Namespace(
        dataset="generated_visits",
        task="mortality",
        batch_size=8,
        epochs=args.epochs,
        patience=10,
        lr=3e-4,
        weight_decay=1e-4,
        embedding_dim=64,
        hidden_dim=64,
        num_heads=4,
        num_htt_layers=1,
        num_rgcn_layers=2,
        dropout=0.1,
        fusion_strategy="cross_attention",
        selection_metric="auroc",
        output_tag="smoke",
    )
    relation_vocab = {"relation_to_id": {str(i): i for i in range(args.relation_count)}}
    summary = scope["run_single_seed"](
        args=train_args,
        seed=20261029,
        device=torch.device("cpu"),
        task_root=out,
        relation_vocab=relation_vocab,
        concept_vocab=vocab,
        train_dataset=datasets[0],
        valid_dataset=datasets[1],
        test_dataset=datasets[2],
        base_suffix="smoke",
    )
    model = captured["model"]
    changed = [
        k
        for k, v in model.named_parameters()
        if not torch.equal(v, captured["before"][k])
    ]
    groups = [
        "ehr_encoder.",
        "pkg_encoder.",
        "cross_attention.",
        "concat_fusion.",
        "prediction_head.",
    ]
    updated = {prefix: any(k.startswith(prefix) for k in changed) for prefix in groups}
    assert all(updated.values()), updated
    batch = next(iter(loaders[2]))
    model.eval()
    with torch.no_grad():
        predictions = model(batch)["logits"]
    fresh = MedCPIIntegrateModel(**config)
    checkpoint = torch.load(
        summary["best_checkpoint_path"], map_location="cpu", weights_only=True
    )
    fresh.load_state_dict(checkpoint["model_state_dict"], strict=True)
    fresh.eval()
    with torch.no_grad():
        repeated = fresh(batch)["logits"]
    equal = bool(torch.equal(predictions, repeated))
    assert equal and bool(torch.isfinite(predictions).all())
    result = dict(
        status="passed",
        data_kind=(
            "synthetic_patients_with_public_DDB_graph"
            if args.input_records
            else "entirely_synthetic"
        ),
        training_runner="author_run_single_seed",
        report_fix="Removed two duplicate fields referring to undefined seed_results; parameter counts use the existing local variables.",
        selection_metric="validation_auroc",
        seed=20261029,
        device="cpu",
        threads=2,
        examples=dict(train=32, validation=8, test=8),
        network=config,
        parameters=sum(p.numel() for p in model.parameters()),
        updated_parameter_groups=updated,
        checkpoint_reloaded_predictions_equal=equal,
        output_shape=list(predictions.shape),
        training=summary,
        test=summary["test_metrics"],
        author_files_verified=len(
            json.loads((ROOT / "source_manifest.json").read_text())["files"]
        ),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
