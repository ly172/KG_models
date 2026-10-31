# Run KAT-GNN on small generated graphs.

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

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
    args = parser.parse_args()
    author = fetch_author()
    sys.path.insert(0, str(author))
    import torch
    from torch_geometric.data import Data, Batch
    from models.gcn_model import GCNModel
    from train.loop import train_one_epoch, validation, test, process_batch
    from utils.checkpoint import save_checkpoint

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20261029)
    random.seed(20261029)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    rng = random.Random(20261029)
    config = SimpleNamespace(
        fusion=True,
        use_pooling=False,
        device=torch.device("cpu"),
        epochs=args.epochs,
        checkpoint="smoke",
        seed=20261029,
    )
    input_dims = {"ccs_graph": 278, "lab_graph": 802}

    def make_graph(patient, view):
        categories = input_dims[view] - 2
        count = 10 + patient % 5
        visits = 2 + patient % 3
        feature_ids = rng.sample(range(categories), count)
        x = torch.zeros((count + visits, categories + 2))
        for i, category in enumerate(feature_ids):
            x[i, category] = 1
        x[count:, -2] = (40 + patient % 46) / 100.0
        x[count:, -1] = (patient // 2) % 2
        edges, weights = [], []

        def link(a, b, w):
            edges.extend([[a, b], [b, a]])
            weights.extend([w, w])

        for v in range(visits):
            for c in rng.sample(range(count), 4):
                link(count + v, c, 1.0)
            if v:
                link(count + v - 1, count + v, 1.0)
        for c in range(count - 1):
            link(c, c + 1, 0.8)
        lags = torch.tensor([180.0 * (visits - v) for v in range(visits)])
        return Data(
            x=x,
            edge_index=torch.tensor(edges).T.contiguous(),
            edge_weight=torch.tensor(weights),
            edge_attr=lags,
            visit_times=torch.tensor([visits]),
        )

    records = []
    for patient in range(48):
        records.append(
            {**{v: make_graph(patient, v) for v in input_dims}, "y": patient % 2}
        )

    def batches(rows):
        result = []
        for start in range(0, len(rows), 8):
            subset = rows[start : start + 8]
            result.append(
                {
                    **{
                        view: Batch.from_data_list([p[view] for p in subset])
                        for view in input_dims
                    },
                    "y": torch.tensor([p["y"] for p in subset]),
                }
            )
        return result

    loaders = [batches(records[:32]), batches(records[32:40]), batches(records[40:])]
    model = GCNModel(input_dims=input_dims, num_classes=2, config=config)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=args.epochs, steps_per_epoch=len(loaders[0])
    )
    criterion = torch.nn.CrossEntropyLoss()
    history = []

    def recorded_train(*values, **options):
        loss, metrics = train_one_epoch(*values, **options)
        history.append(dict(epoch=len(history) + 1, train_loss=loss, train=metrics))
        return loss, metrics

    def recorded_validation(*values, **options):
        loss, metrics = validation(*values, **options)
        history[-1].update(validation_loss=loss, validation=metrics)
        print(json.dumps(history[-1]), flush=True)
        return loss, metrics

    source = (author / "private_all.py").read_text(encoding="utf-8")
    run_function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    first = next(
        i
        for i, n in enumerate(run_function.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "best_val_auprc" for t in n.targets)
    )
    outer_loop = ast.Module(body=run_function.body[first : first + 3], type_ignores=[])
    outer_loop_hash = hashlib.sha256(
        ast.dump(outer_loop, include_attributes=False).encode()
    ).hexdigest()
    scope = dict(
        torch=torch,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        logger=None,
        train_loader=loaders[0],
        valid_loader=loaders[1],
        test_loader=loaders[2],
        train_one_epoch=recorded_train,
        validation=recorded_validation,
        save_checkpoint=save_checkpoint,
        test=test,
    )
    os.chdir(ROOT)
    print("Training KAT-GNN and selecting the best validation AUPRC", flush=True)
    exec(compile(outer_loop, str(author / "private_all.py"), "exec"), scope)
    checkpoint = Path(scope["checkpoint_path"])
    changed = [k for k, v in model.named_parameters() if not torch.equal(v, before[k])]
    gradients = {
        k: float(v.grad.detach().norm())
        for k, v in model.named_parameters()
        if v.grad is not None
    }
    test_metrics = scope["metrics"]
    model.eval()
    with torch.no_grad():
        predictions, _ = process_batch(loaders[2][0], model, config)
    fresh = GCNModel(input_dims=input_dims, num_classes=2, config=config)
    fresh.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    )
    fresh.eval()
    with torch.no_grad():
        repeated, _ = process_batch(loaders[2][0], fresh, config)
    reload_equal = bool(torch.equal(predictions, repeated))
    assert changed and reload_equal and bool(torch.isfinite(predictions).all())
    parameter_groups = ["fusion_weights", "fc."]
    for view in input_dims:
        parameter_groups += [f"convs.{view}.{layer}." for layer in (0, 3, 6)]
        parameter_groups += [f"local_att.{view}.", f"global_att.{view}."]
    updated_groups = {
        prefix: any(k.startswith(prefix) for k in changed)
        for prefix in parameter_groups
    }
    assert all(updated_groups.values()), updated_groups
    result = dict(
        status="passed",
        data_kind="entirely_synthetic",
        outer_training_loop="executed directly from fixed author private_all.run AST",
        outer_loop_sha256=outer_loop_hash,
        selection_metric="validation_auprc",
        seed=20261029,
        device="cpu",
        threads=2,
        epochs=args.epochs,
        examples=dict(train=32, validation=8, test=8),
        input_dimensions=input_dims,
        parameters=sum(p.numel() for p in model.parameters()),
        updated_parameter_groups=updated_groups,
        maximum_gradient_norm=max(gradients.values()),
        checkpoint_reloaded_predictions_equal=reload_equal,
        output_shape=list(predictions.shape),
        test=test_metrics,
        history=history,
        author_files_verified=len(
            json.loads((ROOT / "source_manifest.json").read_text())["files"]
        ),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
