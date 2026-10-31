# Run GraphCare on MIMIC histories with the public UMLS subset.

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

for key in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[key] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from downloads import download_file
from prepare_real import prepare_mimic, prepare_knowledge, original_function, digest


def ensure_resources():
    for item in json.loads(
        (ROOT / "derived_resources.json").read_text(encoding="utf-8")
    )["files"]:
        target = ROOT / "resources" / item["path"]
        assert (
            target.stat().st_size == item["bytes"] and digest(target) == item["sha256"]
        ), "Published exact-match table differs from verified source"
    for file in ["graphcare_resources.json", "pyhealth_resources.json"]:
        for item in json.loads((ROOT / file).read_text(encoding="utf-8"))["files"]:
            target = ROOT / "resources" / item["path"]
            download_file(item["url"], target, item["sha256"], item["bytes"])


class Samples:
    def __init__(self, samples):
        self.samples = samples
        self.patient_to_index = {}
        for i, sample in enumerate(samples):
            self.patient_to_index.setdefault(sample["patient_id"], []).append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mimic-root", default=os.environ.get("MIMIC_IV_ROOT"))
    p.add_argument("--asset-cache")
    p.add_argument(
        "--encoder-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    p.add_argument("--text-batch-size", type=int, default=4)
    p.add_argument(
        "--encoded-cache",
        help="Optional checked local text vectors; graph, name order, fixed encoder, shape and byte hashes must all match.",
    )
    p.add_argument("--patients", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--neighbors", type=int, default=4)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if not args.mimic_root:
        p.error(
            "Supply a MIMIC-IV root with --mimic-root, or use the smoke task for the public Demo."
        )
    private = ROOT / ".private"
    private.mkdir(exist_ok=True)
    report_dir = ROOT / "real_verification"
    report_dir.mkdir(exist_ok=True)
    started = time.time()
    ensure_resources()
    for item in json.loads(
        (ROOT / "upstream_extra/source.json").read_text(encoding="utf-8")
    ):
        assert digest(ROOT / "upstream_extra" / item["path"]) == item["sha256"]
    samples, cohort = prepare_mimic(args.mimic_root, args.patients)
    print(
        json.dumps(
            {
                "step": "real_task_prepared",
                "patients": cohort["patients_with_task_samples"],
                "samples": len(samples),
                "positive_labels": cohort["positive_next_admission_labels"],
            }
        ),
        flush=True,
    )
    sys.path.insert(0, str(ROOT / "public_umls"))
    from fetch_assets import ensure_assets

    assets, assets_report = ensure_assets(args.asset_cache)
    graph, category_maps, knowledge = prepare_knowledge(
        samples, assets / "umls", args.neighbors
    )
    (report_dir / "input_coverage.json").write_text(
        json.dumps(dict(cohort=cohort, knowledge=knowledge), indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "step": "real_knowledge_prepared",
                **{
                    k: knowledge[k]
                    for k in [
                        "nodes",
                        "real_source_triples",
                        "relations",
                        "isolated_ehr_anchors",
                    ]
                },
            }
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from sklearn.metrics import roc_auc_score, average_precision_score

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from vector_cache import load_cached, save_manifest

    if args.encoded_cache:
        node_vectors, relation_vectors, vector_report = load_cached(
            args.encoded_cache, graph, args.encoder_dtype
        )
        np.savez_compressed(
            private / "public_name_vectors.npz",
            node=node_vectors.numpy(),
            relation=relation_vectors.numpy(),
        )
    else:
        from encode_names import encode

        node_vectors, relation_vectors, vector_report = encode(
            assets / "model",
            graph,
            private,
            batch_size=args.text_batch_size,
            model_dtype=args.encoder_dtype,
        )
        vector_report["reused_checked_encoded_vectors"] = False
    vector_report["input_text_origin"] = (
        "Author-public UMLS concept and relation names, plus published PyHealth names for unresolved real CCS/ATC categories."
    )
    save_manifest(private, graph, vector_report)
    torch.manual_seed(20261030)
    np.random.seed(20261030)
    random.seed(20261030)
    from prepare_author import prepare, definition_hash

    author, source_checks = prepare()
    sys.path.insert(0, str(author))
    from graphcare_ import GraphCare
    from graphcare import (
        get_dataloader,
        get_mode_and_out_channels_and_loss_func,
        train,
        evaluate,
        label_ehr_nodes,
    )
    from itertools import chain
    from typing import Optional, Tuple, Union, List

    split, split_hash = original_function(
        ROOT / "upstream/graphcare_/splitter.py",
        "split_by_patient",
        dict(
            np=np,
            torch=torch,
            chain=chain,
            Optional=Optional,
            Tuple=Tuple,
            Union=Union,
            List=List,
            SampleDataset=Samples,
        ),
    )
    pad, pad_hash = original_function(
        ROOT / "upstream_extra/data_prepare.py", "pad_and_convert", {"torch": torch}
    )
    node_index = {node["cui"]: i for i, node in enumerate(graph["nodes"])}
    relation_index = {r: i for i, r in enumerate(graph["relations"])}
    edges = [[node_index[a], node_index[b]] for a, r, b in graph["triples"]]
    rels = [relation_index[r] for a, r, b in graph["triples"]]
    G = Data(
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        relation=torch.tensor(rels, dtype=torch.long),
        y=torch.arange(len(node_index)),
        x=node_vectors.clone(),
        num_nodes=len(node_index),
    )
    neighbors = {i: set() for i in range(len(node_index))}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    max_visits = max(len(s["conditions"]) for s in samples)
    for sample in samples:
        visit_nodes = []
        for i in range(len(sample["conditions"])):
            direct = set()
            for field, ns in [
                ("conditions", "CCSCM"),
                ("procedures", "CCSPROC"),
                ("drugs", "ATC"),
            ]:
                direct.update(category_maps[ns][code] for code in sample[field][i])
            expanded = direct | set().union(*(neighbors[n] for n in direct))
            visit_nodes.append(sorted(expanded))
        sample["node_set"] = sorted(set().union(*(set(v) for v in visit_nodes)))
        sample["visit_padded_node"] = pad(visit_nodes, max_visits, len(node_index))
    # Mark every mapped diagnosis, procedure and drug category.
    samples = label_ehr_nodes(
        "mortality",
        samples,
        len(node_index),
        category_maps["CCSCM"],
        category_maps["CCSPROC"],
        category_maps["ATC"],
    )
    splits = split(Samples(samples), [0.8, 0.1, 0.1], seed=528)
    split_records = []
    for subset in splits:
        part = [subset[i] for i in range(len(subset))]
        split_records.append(
            dict(
                samples=len(part),
                patients=len({s["patient_id"] for s in part}),
                positive_labels=sum(s["label"] for s in part),
            )
        )
    patient_sets = [{s["patient_id"] for s in part} for part in splits]
    assert all(
        not patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i)
    )
    loaders = get_dataloader(G, *splits, task="mortality", batch_size=args.batch_size)
    assert all(
        len(loader) > 0 for loader in loaders
    ), "Small split has no complete author batch; use a smaller explicit --batch-size or larger cohort."
    mode, out_channels, loss_function = get_mode_and_out_channels_and_loss_func(
        "mortality", samples
    )
    config = dict(
        num_nodes=len(node_index),
        num_rels=len(relation_index),
        max_visit=max_visits,
        embedding_dim=1536,
        hidden_dim=128,
        out_channels=out_channels,
        layers=3,
        dropout=0.5,
        decay_rate=0.01,
        node_emb=node_vectors,
        rel_emb=relation_vectors,
        freeze=False,
        patient_mode="joint",
        use_alpha=False,
        use_beta=False,
        use_edge_attn=True,
        self_attn=0.0,
        gnn="BAT",
        attn_init=None,
        drop_rate=0.0,
    )
    model = GraphCare(**config)
    before = {n: v.detach().clone() for n, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    device = torch.device("cpu")
    history = []
    for epoch in range(args.epochs):
        loss = train(
            mode, "joint", "BAT", model, device, loaders[0], optimizer, loss_function
        )
        assert torch.isfinite(loss)
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(loss.detach()) / len(loaders[0]),
            }
        )
    metrics = []
    test_probabilities = None
    for i, loader in enumerate(loaders[1:]):
        labels, probs = evaluate(mode, "joint", "BAT", model, device, loader)
        y = np.asarray(labels).reshape(-1)
        pr = np.asarray(probs).reshape(-1)
        assert np.isfinite(pr).all()
        metrics.append(
            dict(
                evaluated_samples=len(y),
                positive_labels=int(y.sum()),
                auroc=float(roc_auc_score(y, pr)) if len(set(y)) == 2 else None,
                average_precision=(
                    float(average_precision_score(y, pr)) if y.sum() > 0 else None
                ),
            )
        )
        if i == 1:
            test_probabilities = pr
    changed = sum(
        not torch.equal(before[n], v.detach()) for n, v in model.named_parameters()
    )
    assert changed > 0
    torch.save(model.state_dict(), private / "real_mimic_model.pt")
    restored = GraphCare(**config)
    restored.load_state_dict(
        torch.load(
            private / "real_mimic_model.pt", map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    _, reloaded = evaluate(mode, "joint", "BAT", restored, device, loaders[2])
    equal = bool(np.array_equal(test_probabilities, np.asarray(reloaded).reshape(-1)))
    assert equal
    for name in [
        "get_subgraph",
        "get_dataloader",
        "Dataset",
        "train",
        "evaluate",
        "label_ehr_nodes",
        "get_mode_and_out_channels_and_loss_func",
    ]:
        original = definition_hash(
            (ROOT / "upstream/graphcare.py").read_text(encoding="utf-8"), name
        )
        runtime = definition_hash(
            (author / "graphcare.py").read_text(encoding="utf-8"), name
        )
        assert original == runtime
        source_checks["graphcare.py:" + name] = dict(original=original, runtime=runtime)
    result = dict(
        status="passed",
        cohort=cohort,
        knowledge=knowledge,
        vector_encoding=vector_report,
        source_checks=source_checks,
        split_function_sha256=split_hash,
        pad_function_sha256=pad_hash,
        splits=split_records,
        batch_size=args.batch_size,
        batches=[len(x) for x in loaders],
        actual_evaluated=[len(x) * args.batch_size for x in loaders],
        epochs=args.epochs,
        optimizer_updates=args.epochs * len(loaders[0]),
        training_history=history,
        evaluation=metrics,
        parameter_tensors_changed=changed,
        strict_checkpoint_reload_identical_predictions=equal,
        graph_encoder="original BAT 3 layers, hidden 128; edge attention enabled",
        configuration_scope="actual author single_run default switches, including alpha=False and beta=False",
        original_total_entry_not_called="Author single_run requires unavailable private files; this runner preserves its actual default model switches and calls original task/train/evaluate.",
        private_patient_and_model_outputs=".private/",
        seconds=round(time.time() - started, 3),
        scope="Author GraphCare prediction network on MIMIC-IV histories with a public UMLS subset and local GTE name vectors.",
    )
    (report_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "optimizer_updates",
                    "parameter_tensors_changed",
                    "strict_checkpoint_reload_identical_predictions",
                    "seconds",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
