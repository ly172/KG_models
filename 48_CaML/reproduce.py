# Run the CaML network and its updates on generated tasks.

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import ssl
import subprocess
import textwrap
import types
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    sources = json.loads((ROOT / "sources.json").read_text())
    import truststore

    for name, source in sources.items():
        target = ROOT / "cache" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) == source["sha256"]:
            continue
        temp = target.with_suffix(target.suffix + ".download")
        print("Downloading", name, flush=True)
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        try:
            with urllib.request.urlopen(
                source["url"], context=context, timeout=120
            ) as response, temp.open("wb") as f:
                shutil.copyfileobj(response, f)
        except Exception:
            if os.name != "nt":
                raise
            subprocess.run(
                [
                    "curl.exe",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--output",
                    str(temp),
                    source["url"],
                ],
                check=True,
            )
        if sha(temp) != source["sha256"]:
            raise RuntimeError("Source checksum differs: " + name)
        temp.replace(target)


def definitions(relative, names, namespace):
    text = (ROOT / "cache" / relative).read_text(encoding="utf-8")
    nodes = [
        n
        for n in ast.parse(text).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)


def native_block(start, end):
    lines = (
        (ROOT / "cache/metalearn/train_caml.py")
        .read_text(encoding="utf-8")
        .splitlines(True)
    )
    text = textwrap.dedent("".join(lines[start - 1 : end]))
    ast.parse(text)
    return compile(text, "author_train_caml_%d_%d" % (start, end), "exec"), text


def main():
    prepare()
    import numpy as np
    import torch
    from torch import nn
    from tqdm import tqdm

    torch.set_num_threads(2)
    torch.manual_seed(23)
    rng = np.random.default_rng(23)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    namespace = {
        "torch": torch,
        "nn": nn,
        "np": np,
        "pickle": pickle,
        "tqdm": tqdm,
        "hashlib": hashlib,
    }
    definitions("src/models/xlearner.py", ["MLPLayer", "MLPLateConcatModel"], namespace)
    definitions("metalearn/task_utils.py", ["TaskEmbeddingMapper"], namespace)
    definitions("metalearn/caml.py", ["CAMLAdapter"], namespace)
    definitions("causal_singletask/consts_and_utils.py", ["get_optimizer"], namespace)
    mapper = namespace["TaskEmbeddingMapper"](
        str(ROOT / "cache/task_embedding/tasks/task_embeddings_dict_stargraph256.pkl")
    )
    drug_ids = list(mapper.d)[:3]
    windows = ["0", "1", "2_7", "8_30", "31_90", "91_365", "over_365"]
    feature_names = [
        f"{kind}_code_{code}_days_{window}"
        for window in windows
        for kind in ["diagnosis", "procedure", "drug"]
        for code in [0, 1]
    ]
    tasks = []
    for drug in drug_ids:
        x = rng.binomial(1, 0.25, size=(32, len(feature_names))).astype("float32")
        embedding = np.asarray(mapper(drug), dtype="float32")
        target = np.tanh(0.3 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * embedding[0]).astype(
            "float32"
        )
        e = np.repeat(embedding[None, :], len(x), axis=0)
        tasks.append(
            (torch.from_numpy(x), torch.from_numpy(e), torch.from_numpy(target))
        )
    model = namespace["MLPLateConcatModel"](
        len(feature_names),
        256,
        1,
        dropout=0.0,
        model_dim=256,
        n_layers=6,
        batch_norm=False,
        regress=True,
        residual=True,
    )
    args = types.SimpleNamespace(
        task_embedding_config="late_concat",
        sin_learner=False,
        graphite_learner=False,
        caml_r_learner=False,
        null_patients2=False,
        l1_reg=1e-7,
        learning_rate=0.001,
        weight_decay=0.005,
    )
    adapter = namespace["CAMLAdapter"](nn.MSELoss(), 1, 5, torch.device("cpu"), args)
    get_optimizer = namespace["get_optimizer"]
    meta_optimizer = torch.optim.SGD(model.parameters(), lr=2.0)
    initial_optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    context = {
        "torch": torch,
        "copy": copy,
        "args": args,
        "model": model,
        "meta_optimizer": meta_optimizer,
        "inner_optimizer_state": initial_optimizer.state_dict(),
        "get_optimizer": get_optimizer,
        "train_adapter": adapter,
        "n_iterations": 2,
        "meta_lr": 2.0,
        "meta_lr_final": 0.2,
        "meta_batch_size": 1,
    }
    begin, begin_text = native_block(365, 375)
    adapt, adapt_text = native_block(570, 588)
    finish, finish_text = native_block(650, 653)
    (out / "executed_author_updates.py").write_text(
        begin_text + "\n" + adapt_text + "\n" + finish_text, encoding="utf-8"
    )
    losses = []
    for iteration in range(2):
        context.update(iteration=iteration, meta_train_loss=0.0)
        exec(begin, context)
        x, e, y = tasks[iteration]
        context["train_loader"] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x[:24], e[:24], y[:24]),
            batch_size=8,
            shuffle=False,
        )
        context["val_loader"] = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x[24:], e[24:], y[24:]),
            batch_size=8,
            shuffle=False,
        )
        exec(adapt, context)
        losses.append(float(context["meta_train_loss"]))
        exec(finish, context)
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    model.eval()
    x, e, y = tasks[2]
    with torch.no_grad():
        predictions = model(x, e)
    assert changed and bool(torch.isfinite(predictions).all())
    torch.save(model.state_dict(), out / "model.pt")
    reloaded = namespace["MLPLateConcatModel"](
        len(feature_names),
        256,
        1,
        dropout=0.0,
        model_dim=256,
        n_layers=6,
        batch_norm=False,
        regress=True,
        residual=True,
    )
    reloaded.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        repeated = reloaded(x, e)
    assert torch.equal(predictions, repeated)
    np.savez(
        out / "generated_tasks.npz",
        **{
            f"{i}_{name}": tensor.numpy()
            for i, task in enumerate(tasks)
            for name, tensor in zip(["history", "drug_embedding", "target"], task)
        },
    )
    report = {
        "status": "passed",
        "author_commit": "88e2693ffa6996f8768b1eaa2b96f2a600643328",
        "scope": "author network and original inner/outer updates on synthetic claims-shaped tasks",
        "patient_data": "generated code occurrences in the author seven time windows",
        "target_formula": "tanh(0.3*history[0]-0.2*history[1]+0.1*drug_embedding[0])",
        "feature_names": feature_names,
        "drug_ids": drug_ids,
        "public_drug_vectors": 744,
        "model": "MLPLateConcatModel, model_dim=256, n_layers=6, residual=True, batch_norm=False",
        "meta_iterations": 2,
        "meta_batch_size": 1,
        "inner_updates_per_task": 1,
        "updated_parameter_tensors": len(changed),
        "validation_losses": losses,
        "held_out_task": drug_ids[2],
        "prediction_shape": list(predictions.shape),
        "author_update_ranges": [[365, 375], [570, 588], [650, 653]],
    }
    (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
