# Reuse cached name vectors when the graph and encoder match.

import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ENCODER = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
REVISION = "a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644"


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_identity(graph, model_dtype="float32"):
    texts = [n["name"] for n in graph["nodes"]] + [
        r.replace("_", " ") for r in graph["relations"]
    ]
    return dict(
        graph_sha256=json_hash(graph),
        ordered_texts_sha256=json_hash(texts),
        concepts=len(graph["nodes"]),
        relations=len(graph["relations"]),
        width=1536,
        encoder=ENCODER,
        revision=REVISION,
        max_tokens=128,
        pooling="last unpadded token, followed by L2 normalization",
        attention="eager, bidirectional",
        dtype="float32",
        model_dtype=model_dtype,
    )


def save_manifest(directory, graph, report):
    directory = Path(directory)
    assert report["encoder"] == ENCODER and report["revision"] == REVISION
    values = np.load(directory / "public_name_vectors.npz")
    full = np.concatenate([values["node"], values["relation"]], axis=0)
    vector_hash = hashlib.sha256(full.tobytes()).hexdigest()
    assert vector_hash == report["vector_bytes_sha256"]
    record = dict(
        identity=input_identity(graph, report.get("model_dtype", "float32")),
        npz_sha256=file_hash(directory / "public_name_vectors.npz"),
        vector_bytes_sha256=vector_hash,
        encoding_report=report,
    )
    (directory / "name_vectors_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def load_cached(directory, graph, model_dtype="float32"):
    directory = Path(directory)
    record = json.loads(
        (directory / "name_vectors_manifest.json").read_text(encoding="utf-8")
    )
    record["identity"].setdefault("model_dtype", "float32")
    if record["identity"] != input_identity(graph, model_dtype):
        raise ValueError(
            "Cached vector graph, text order, or fixed encoder identity does not match."
        )
    for key in ["encoder", "revision", "max_tokens", "pooling", "attention", "dtype"]:
        if record["encoding_report"][key] != record["identity"][key]:
            raise ValueError(
                "Cached encoder report contradicts its fixed identity: " + key
            )
    if record["encoding_report"].get("model_dtype", "float32") != model_dtype:
        raise ValueError(
            "Cached vectors used a different encoder arithmetic precision."
        )
    if (
        record["encoding_report"]["concepts"] != len(graph["nodes"])
        or record["encoding_report"]["relations"] != len(graph["relations"])
        or record["encoding_report"]["output_width"] != 1536
    ):
        raise ValueError("Cached encoder report dimensions do not match graph.")
    path = directory / "public_name_vectors.npz"
    if file_hash(path) != record["npz_sha256"]:
        raise ValueError("Cached vector file SHA256 does not match manifest.")
    values = np.load(path)
    nodes, relations = values["node"], values["relation"]
    if nodes.shape != (len(graph["nodes"]), 1536) or relations.shape != (
        len(graph["relations"]),
        1536,
    ):
        raise ValueError("Cached vector shapes do not match graph.")
    full = np.concatenate([nodes, relations], axis=0)
    if full.dtype != np.float32 or not np.isfinite(full).all():
        raise ValueError("Cached vectors must be finite float32 values.")
    value_hash = hashlib.sha256(full.tobytes()).hexdigest()
    if (
        value_hash != record["vector_bytes_sha256"]
        or value_hash != record["encoding_report"]["vector_bytes_sha256"]
    ):
        raise ValueError(
            "Cached vector bytes do not match independently recorded encoder output."
        )
    if not np.allclose(np.linalg.norm(full, axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError(
            "Cached vector normalization differs from fixed encoder procedure."
        )
    report = dict(record["encoding_report"])
    report["reused_checked_encoded_vectors"] = True
    report["cache_graph_sha256"] = record["identity"]["graph_sha256"]
    report["cache_ordered_texts_sha256"] = record["identity"]["ordered_texts_sha256"]
    report["cache_npz_sha256"] = record["npz_sha256"]
    return torch.from_numpy(nodes.copy()), torch.from_numpy(relations.copy()), report
