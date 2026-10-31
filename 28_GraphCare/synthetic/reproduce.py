# Run GraphCare on artificial inputs.

import argparse
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

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from prepare_author import ROOT, prepare
from simulate_inputs import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20261029)
    np.random.seed(20261029)
    random.seed(20261029)
    author, source_checks = prepare()
    sys.path.insert(0, str(author))
    from graphcare_ import GraphCare
    from graphcare import train, evaluate, get_mode_and_out_channels_and_loss_func

    samples, node_embedding, relation_embedding, summary = generate()
    loaders = [
        DataLoader(
            part, batch_size=64, shuffle=index == 0, drop_last=True, num_workers=0
        )
        for index, part in enumerate([samples[:128], samples[128:192], samples[192:]])
    ]
    mode, out_channels, loss_function = get_mode_and_out_channels_and_loss_func(
        "mortality", samples
    )
    device = torch.device("cpu")
    model = GraphCare(
        num_nodes=32,
        num_rels=5,
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
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    result = {
        "status": "running",
        "model": "Author GraphCare (alpha=False, beta=False)",
        "inputs": summary,
        "core_source_checks": source_checks,
        "settings": {
            "hidden_dim": 128,
            "layers": 3,
            "dropout": 0.5,
            "decay_rate": 0.01,
            "batch_size": 64,
            "epochs": args.epochs,
            "optimizer_steps_per_epoch": 2,
            "patient_mode": "joint",
            "gnn": "BAT",
            "alpha": False,
            "beta": False,
            "edge_attention": True,
            "loss": "author binary_cross_entropy_with_logits",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
        },
        "epochs": [],
    }
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    start = time.time()
    try:
        for epoch in range(args.epochs):
            total_loss = train(
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
                "mean_training_loss": float(total_loss.detach()) / len(loaders[0]),
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
            not torch.equal(initial[name], value.detach())
            for name, value in model.named_parameters()
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
        result["seconds"] = round(time.time() - start, 3)
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
