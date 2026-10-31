# Run GreaseLM training on a small MedQA subset.

import argparse
import ast
import difflib
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["INHERIT_BERT"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected):
    sys.path.insert(0, str(ROOT.parent / "shared"))
    from downloads import download_file

    download_file(url, path, expected)


def prepare(counts):
    manifest = json.loads((ROOT / "author_source.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(ROOT / "author" / name) != expected:
            raise RuntimeError("Original author source changed: " + name)
    for name, resource in json.loads(
        (ROOT / "sapbert_source.json").read_text()
    ).items():
        download(resource["url"], ROOT / "assets/sapbert" / name, resource["sha256"])
    archive = ROOT / "assets/data_preprocessed_biomed.zip"
    download(
        "https://drive.usercontent.google.com/download?id=1WtVE-6iffn3O6Gsgrf9E-ZBf9hKpGgDl&export=download&confirm=t",
        archive,
        "9d1d5508e1e6afe575cf5a94ddefa5fd2c43703d230d2db327b0d5bff1171e45",
    )
    work = ROOT / "work"
    paths = {}
    with zipfile.ZipFile(archive) as bundle:

        def extract(relative):
            names = [
                n
                for n in bundle.namelist()
                if n == relative or n.endswith("/" + relative)
            ]
            if len(names) != 1:
                raise RuntimeError("Unexpected archive member: " + relative)
            target = work / "data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(names[0]) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            return target

        extract("ddb/vocab.txt")
        extract("ddb/ent_emb.npy")
        for split, count in counts.items():
            statement = extract("medqa_usmle/statement/" + split + ".statement.jsonl")
            adjacency = extract("medqa_usmle/graph/" + split + ".graph.adj.pk")
            lines = statement.read_text("utf-8").splitlines()[:count]
            if len(lines) != count:
                raise ValueError("Requested subset exceeds split size")
            choices = len(json.loads(lines[0])["question"]["choices"])
            entries = []
            required = count * choices
            with adjacency.open("rb") as source:
                while len(entries) < required:
                    try:
                        part = pickle.load(source)
                    except EOFError:
                        break
                    entries.extend(
                        part[: required - len(entries)]
                        if isinstance(part, list)
                        else [part]
                    )
                    del part
            if len(entries) != required:
                raise ValueError(
                    "The author graph file has fewer entries than the selected questions require"
                )
            statement.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with adjacency.open("wb") as dest:
                pickle.dump(entries[: count * choices], dest)
            paths[split] = {
                "questions": count,
                "statement_sha256": sha(statement),
                "graph_sha256": sha(adjacency),
                "selection": "first records in author split",
            }
            for parent in (statement.parent, adjacency.parent):
                for cache in parent.glob("*.loaded_cache"):
                    cache.unlink()
    return work, paths


def patched_entry(work):
    original = (ROOT / "author/greaselm.py").read_text("utf-8")
    deletion = '            del model_state_dict["lmgnn.concept_emb.emb.weight"]'
    final_call = "    main(args)"
    assert original.count(deletion) == original.count(final_call) == 1
    patched = original.replace(
        deletion, "            # Keep the actual frozen table in the saved state."
    )
    patched = patched.replace(final_call, "    run_selected(args, globals())")
    # The removed deletion is the only change inside train().
    original_tree, patched_tree = ast.parse(original), ast.parse(patched)
    a = next(
        x
        for x in original_tree.body
        if isinstance(x, ast.FunctionDef) and x.name == "train"
    )
    b = next(
        x
        for x in patched_tree.body
        if isinstance(x, ast.FunctionDef) and x.name == "train"
    )

    class RemoveCheckpointDeletion(ast.NodeTransformer):
        def visit_Delete(self, node):
            if ast.dump(node).find("model_state_dict") >= 0:
                return None
            return node

    a = RemoveCheckpointDeletion().visit(a)
    assert ast.dump(a, include_attributes=False) == ast.dump(
        b, include_attributes=False
    )
    path = work / "greaselm_entry.py"
    path.write_text(patched, encoding="utf-8")
    (ROOT / "outputs/checkpoint_and_entry.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                patched.splitlines(True),
                fromfile="author/greaselm.py",
                tofile="work/greaselm_entry.py",
            )
        ),
        encoding="utf-8",
    )
    return path


def run_selected(args, namespace):
    import numpy as np
    import torch
    import wandb
    from utils import data_utils

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    encoder = str(ROOT / "assets/sapbert")
    data_utils.MODEL_NAME_TO_CLASS[encoder] = "bert"
    devices = (torch.device("cpu"), torch.device("cpu"))
    captured = {}
    original_load = namespace["load_data"]
    original_construct = namespace["construct_model"]

    def load(*values, **options):
        data = original_load(*values, **options)
        captured["data"] = data
        return data

    def construct(*values, **options):
        model = original_construct(*values, **options)
        captured["model"] = model
        captured["initial_parameter_hashes"] = {
            n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n, p in model.named_parameters()
        }
        return model

    namespace["load_data"] = load
    namespace["construct_model"] = construct
    print("Starting original author training procedure", flush=True)
    with wandb.init(mode="disabled"):
        namespace["train"](args, False, True, devices, "ddb")
    model = captured["model"]
    data = captured["data"]
    changed = [
        n
        for n, p in model.named_parameters()
        if captured["initial_parameter_hashes"][n]
        != hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
    ]
    assert changed, "No learned parameter changed"
    model.eval()
    batch = next(iter(data.test()))
    with torch.no_grad():
        scores, _ = model(*batch[2:])
    checkpoint = Path(args.save_dir) / ("model.pt." + str(args.n_epochs - 1))
    saved = torch.load(str(checkpoint), map_location="cpu")
    key = "lmgnn.concept_emb.emb.weight"
    assert key in saved["model"]
    training_table_equal = bool(
        torch.equal(saved["model"][key], model.state_dict()[key])
    )
    saved.pop("optimizer", None)
    saved.pop("scheduler", None)
    model.zero_grad(set_to_none=True)
    del captured["model"], model
    gc.collect()
    # Check the load error caused by the omitted concept table.
    omitted = {k: v for k, v in saved["model"].items() if k != key}
    fresh = original_construct(args, "ddb")
    fresh.lmgnn.mp.resize_token_embeddings(len(data.tokenizer))
    try:
        fresh.load_state_dict(omitted, strict=True)
    except RuntimeError as error:
        omission_error = str(error)
    else:
        raise AssertionError("Expected original missing-state error")
    loading = fresh.load_state_dict(saved["model"], strict=True)
    model = fresh
    fresh.eval()
    with torch.no_grad():
        repeated, _ = fresh(*batch[2:])
    difference = float((scores - repeated).abs().max())
    assert torch.equal(scores, repeated) and bool(torch.isfinite(scores).all())
    frozen = model.lmgnn.concept_emb.emb.weight.detach().cpu().numpy()
    raw = np.load("data/ddb/ent_emb.npy", mmap_mode="r")
    report = {
        "status": "passed",
        "data_kind": "real_author_processed_MedQA",
        "checkpoint_change": "Retain the actual frozen concept table deleted by original saving code.",
        "epochs": args.n_epochs,
        "optimizer_updates": saved["global_step"],
        "counts": {
            "train": data.train_size(),
            "dev": data.dev_size(),
            "test": data.test_size(),
        },
        "original_architecture": {
            "k": args.k,
            "gnn_dim": args.gnn_dim,
            "ie_dim": args.ie_dim,
            "max_seq_len": args.max_seq_len,
            "max_node_num": args.max_node_num,
        },
        "parameter_tensors_updated": len(changed),
        "updated_names": changed,
        "original_omission_strict_load_error": omission_error,
        "full_state_missing_keys": loading.missing_keys,
        "full_state_unexpected_keys": loading.unexpected_keys,
        "max_reload_difference": difference,
        "test_batch_shape": list(scores.shape),
        "frozen_concept_table_sha256": hashlib.sha256(frozen.tobytes()).hexdigest(),
        "frozen_table_equals_training_table": training_table_equal,
        "original_initialization_max_difference_from_supplied_vectors": float(
            np.abs(frozen[: len(raw)] - raw).max()
        ),
        "initialization_note": "The author initialization writes new values to the supplied frozen concept table.",
        "checkpoint_sha256": sha(checkpoint),
        "training_log": "outputs/native/log.csv",
        "scope": "Small training and checkpoint round trip on author MedQA inputs.",
    }
    (ROOT / "outputs/result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "updated_names"}, indent=2),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument(
        "--auto-sequence-length",
        action="store_true",
        help="Pad to the longest complete selected question and answer pair",
    )
    options = parser.parse_args()
    if min(options.train_count, options.eval_count, options.epochs) < 1:
        parser.error("Counts and epochs must be positive")
    (ROOT / "outputs").mkdir(exist_ok=True)
    started = time.time()
    work, inputs = prepare(
        {
            "train": options.train_count,
            "dev": options.eval_count,
            "test": options.eval_count,
        }
    )
    sequence_length = 512
    if options.auto_sequence_length:
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            str(ROOT / "assets/sapbert"), local_files_only=True
        )
        lengths = []
        for split in ("train", "dev", "test"):
            for line in (
                (work / "data/medqa_usmle/statement" / (split + ".statement.jsonl"))
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                record = json.loads(line)
                context = record["question"]["stem"]
                if "para" in record:
                    context = record["para"] + " " + context
                if "fact1" in record:
                    context = record["fact1"] + " " + context
                for choice in record["question"]["choices"]:
                    lengths.append(
                        len(
                            tokenizer(context, " " + choice["text"], truncation=False)[
                                "input_ids"
                            ]
                        )
                    )
        sequence_length = max(lengths)
        if sequence_length > 512:
            raise ValueError(
                "A selected complete question exceeds the encoder limit of 512 tokens."
            )
        inputs["text_lengths"] = {
            "maximum_complete_pair": sequence_length,
            "truncated_pairs": 0,
        }
    (ROOT / "outputs/input_provenance.json").write_text(
        json.dumps(inputs, indent=2), encoding="utf-8"
    )
    entry = patched_entry(work)
    os.chdir(work)
    sys.path.insert(0, str(ROOT / "author"))
    sys.argv = [
        str(entry),
        "--dataset",
        "medqa_usmle",
        "--encoder",
        str(ROOT / "assets/sapbert"),
        "-k",
        "3",
        "--gnn_dim",
        "200",
        "-elr",
        "5e-5",
        "-dlr",
        "1e-3",
        "-bs",
        "128",
        "--seed",
        "5",
        "-mbs",
        str(options.micro_batch),
        "--eval_batch_size",
        "1",
        "--unfreeze_epoch",
        "0",
        "--encoder_layer=-1",
        "-sl",
        str(sequence_length),
        "--max_node_num",
        "200",
        "--n_epochs",
        str(options.epochs),
        "--max_epochs_before_stop",
        "10",
        "--save_dir",
        str(ROOT / "outputs/native"),
        "--run_name",
        "greaselm_original_small_medqa",
        "--ie_dim",
        "400",
        "--info_exchange",
        "true",
        "--ie_layer_num",
        "1",
        "--sep_ie_layers",
        "false",
        "--random_ent_emb",
        "false",
        "--ent_emb",
        "ddb",
        "--lr_schedule",
        "fixed",
        "--data_dir",
        "data",
        "--cuda",
        "false",
    ]
    runpy.run_path(
        str(entry), init_globals={"run_selected": run_selected}, run_name="__main__"
    )
    print("Finished in %.1f seconds" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
