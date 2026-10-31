# Extract one OpenBookQA question and its graphs from the author's data.

from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="The data folder extracted from the public author archive",
    )
    args = parser.parse_args()
    root = args.data_root
    files = {
        name: root / name
        for name in [
            "obqa/statement/train.statement.jsonl",
            "obqa/graph/train.graph.adj.pk",
            "cpnet/tzw.ent.npy",
        ]
    }
    provenance = {
        "archive_url": "https://drive.google.com/file/d/1w_0x9BxQ2xvdl-PkBP5AU5VusEFQecTu/view",
        "archive_sha256": "41e5304230a533c3e6901fa7d3e400b524a55e5f9fd63f1ba8a1740b62e154ef",
        "sources": {
            name: {"sha256": sha(p), "bytes": p.stat().st_size}
            for name, p in files.items()
        },
        "selection": "First training question, original first four choice graphs; embeddings for the concept rows used under the original 200-node limit",
    }
    expected = ROOT / "data/provenance.json"
    if expected.exists():
        previous = json.loads(expected.read_text())
        if previous["sources"] != provenance["sources"]:
            raise RuntimeError(
                "The public source files differ from the recorded version"
            )
    question = json.loads(
        files["obqa/statement/train.statement.jsonl"].open(encoding="utf-8").readline()
    )
    with files["obqa/graph/train.graph.adj.pk"].open("rb") as f:
        graphs = pickle.load(f)[:4]
    raw = []
    for graph in graphs:
        adj = graph["adj"].tocoo()
        raw.append(
            {
                "concepts": graph["concepts"].tolist(),
                "qmask": graph["qmask"].tolist(),
                "amask": graph["amask"].tolist(),
                "cid2score": [
                    [int(c), float(s)] for c, s in graph["cid2score"].items()
                ],
                "adj": {
                    "shape": list(adj.shape),
                    "row": adj.row.tolist(),
                    "col": adj.col.tolist(),
                    "data": adj.data.tolist(),
                },
            }
        )
    used = sorted({0} | {int(c) for graph in graphs for c in graph["concepts"][:199]})
    original = np.load(files["cpnet/tzw.ent.npy"], mmap_mode="r")
    embeddings = np.array(original[used], dtype=np.float32)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    np.save(out / "concept_embeddings.npy", embeddings)
    (out / "example.json").write_text(
        json.dumps(
            {"question": question, "graphs": raw, "embedding_concept_ids": used}
        ),
        encoding="utf-8",
    )
    provenance.update(
        {
            "question_id": question["id"],
            "graphs": 4,
            "embedding_rows": len(used),
            "embedding_dimension": 1024,
            "files": {
                name: sha(out / name)
                for name in ["example.json", "concept_embeddings.npy"]
            },
        }
    )
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
