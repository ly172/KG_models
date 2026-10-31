# Run GraphCare with public UMLS, local vectors and artificial visits.

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

for name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from fetch_assets import ROOT, ensure_assets
from public_inputs import select_graph, make_simulated_samples
from encode_names import encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-cache",
        help="Optional verified public download cache; generated vectors are always rebuilt",
    )
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    started = time.time()
    assets, asset_report = ensure_assets(args.asset_cache)
    graph = select_graph(assets / "umls")
    (output / "selected_public_graph.json").write_text(
        json.dumps(graph, indent=2), encoding="utf-8"
    )
    node_embedding, relation_embedding, vector_report = encode(
        assets / "model", graph, output
    )
    # Reset the seed after encoding names.
    torch.manual_seed(20261029)
    np.random.seed(20261029)
    random.seed(20261029)
    parent = ROOT.parent
    sys.path.insert(0, str(parent))
    from prepare_author import prepare, definition_hash

    author, checks = prepare()
    sys.path.insert(0, str(author))
    from graphcare_ import GraphCare
    from graphcare import (
        get_dataloader,
        get_mode_and_out_channels_and_loss_func,
        train,
        evaluate,
    )

    for name in ["get_subgraph", "get_dataloader", "Dataset"]:
        original = definition_hash(
            (parent / "upstream/graphcare.py").read_text(encoding="utf-8"), name
        )
        runtime = definition_hash(
            (author / "graphcare.py").read_text(encoding="utf-8"), name
        )
        assert original == runtime
        checks["graphcare.py:" + name] = {
            "original": original,
            "runtime": runtime,
        }
    G, splits, sample_report = make_simulated_samples(graph, node_embedding)
    loaders = get_dataloader(G, *splits, task="mortality", batch_size=64)
    mode, out_channels, loss_function = get_mode_and_out_channels_and_loss_func(
        "mortality", splits[0]
    )
    device = torch.device("cpu")
    model = GraphCare(
        num_nodes=len(graph["nodes"]),
        num_rels=len(graph["relations"]),
        max_visit=3,
        embedding_dim=1536,
        hidden_dim=128,
        out_channels=out_channels,
        layers=3,
        dropout=0.5,
        decay_rate=0.01,
        node_emb=node_embedding,
        rel_emb=relation_embedding,
        freeze=False,
        patient_mode="joint",
        use_alpha=False,
        use_beta=False,
        use_edge_attn=True,
        self_attn=0.0,
        gnn="BAT",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    initial = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    result = {
        "status": "running",
        "model": "Author GraphCare (alpha=False, beta=False)",
        "input_kind": "author_public_UMLS_and_local_GTE_vectors_with_synthetic_patient_visits",
        "assets": asset_report,
        "graph_summary": {
            key: value
            for key, value in graph.items()
            if key not in ["nodes", "relations", "triples"]
        },
        "node_count": len(graph["nodes"]),
        "relation_count": len(graph["relations"]),
        "vector_encoding": vector_report,
        "simulated_visits": sample_report,
        "core_source_checks": checks,
        "settings": {
            "layers": 3,
            "hidden_dimension": 128,
            "input_dimension": 1536,
            "batch_size": 64,
            "dropout": 0.5,
            "decay_rate": 0.01,
            "patient_mode": "joint",
            "gnn": "BAT",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "epochs": args.epochs,
            "updates_per_epoch": len(loaders[0]),
        },
        "epochs": [],
    }
    try:
        for epoch in range(args.epochs):
            loss = train(
                mode,
                "joint",
                "BAT",
                model,
                device,
                loaders[0],
                optimizer,
                loss_function,
            )
            labels, probabilities = evaluate(
                mode, "joint", "BAT", model, device, loaders[1]
            )
            values = {
                "epoch": epoch + 1,
                "mean_training_loss": float(loss.detach()) / len(loaders[0]),
                "artificial_validation_auroc": float(
                    roc_auc_score(labels, probabilities)
                ),
                "artificial_validation_auprc": float(
                    average_precision_score(labels, probabilities)
                ),
            }
            assert all(math.isfinite(value) for value in values.values())
            result["epochs"].append(values)
        labels, probabilities = evaluate(
            mode, "joint", "BAT", model, device, loaders[2]
        )
        result["artificial_test_metrics"] = {
            "auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
        }
        result["updated_parameter_tensors"] = sum(
            not torch.equal(initial[name], parameter.detach())
            for name, parameter in model.named_parameters()
        )
        result["finite_gradients"] = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        assert result["updated_parameter_tensors"] > 0 and result["finite_gradients"]
        torch.save(model.state_dict(), output / "model.pt")
        saved_probabilities = np.asarray(probabilities).copy()
        model.load_state_dict(
            torch.load(output / "model.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        _, repeated_probabilities = evaluate(
            mode, "joint", "BAT", model, device, loaders[2]
        )
        result["checkpoint_reloaded_predictions_equal"] = bool(
            np.array_equal(saved_probabilities, np.asarray(repeated_probabilities))
        )
        assert result["checkpoint_reloaded_predictions_equal"]
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        result["seconds"] = round(time.time() - started, 3)
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
