# Run the authors' AIFB preparation and training scripts.

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
COMMIT = "4bec1341dd46b72bf482f7ed26c2dca4533577f6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hash(expected):
    return lambda path: sha(path) == expected


def archive_content(expected):
    def check(path):
        try:
            with zipfile.ZipFile(path) as bundle:
                rows = sorted(
                    [
                        info.filename.split("/", 1)[1],
                        hashlib.sha256(bundle.read(info)).hexdigest(),
                    ]
                    for info in bundle.infolist()
                    if not info.is_dir()
                )
        except (zipfile.BadZipFile, IndexError):
            return False
        return (
            hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
            == expected
        )

    return check


def download(url, path, matches):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and matches(path):
        return
    print("Downloading", path.name, flush=True)
    temp = path.with_suffix(path.suffix + ".download")
    if os.name == "nt":
        subprocess.run(
            [
                "curl.exe",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--output",
                str(temp),
                url,
            ],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, temp)
    if not matches(temp):
        raise RuntimeError("Resource checksum differs: " + path.name)
    temp.replace(path)


def prepare():
    archive = ROOT / "cache/author.zip"
    download(
        "https://codeload.github.com/tkipf/relational-gcn/zip/" + COMMIT,
        archive,
        archive_content(
            "92f32041b9675c144bbbe8bf6db70a1924030bebf7f3c1d6db09fce2b5c452e6"
        ),
    )
    graph = ROOT / "cache/aifb_stripped.nt.gz"
    download(
        "https://www.dropbox.com/s/fkvgvkygo2gf28k/aifb_stripped.nt.gz?dl=1",
        graph,
        file_hash("ba4191235456e09d8f12a2415f24273b8d7641ffee6001ca9340f02c04d77fea"),
    )
    work = ROOT / "work"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(work)
    source = work / ("relational-gcn-" + COMMIT)
    shutil.copy2(graph, source / "rgcn/data/aifb/aifb_stripped.nt.gz")
    changes = []
    replacements = {
        "rgcn/data_utils.py": [
            ("nodes_dict.iteritems()", "nodes_dict.items()"),
            ("isinstance(input, unicode)", "isinstance(input, str)"),
        ],
        "rgcn/prepare_dataset.py": [
            ("rel_list = range(len(A))", "rel_list = list(range(len(A)))"),
            ("rel_dict.iteritems()", "rel_dict.items()"),
            ("bfs_generator.next()", "next(bfs_generator)"),
        ],
        "rgcn/utils.py": [("len(train_idx) / 5", "len(train_idx) // 5")],
    }
    hashes = {}
    for p in source.rglob("*.py"):
        hashes[str(p.relative_to(source)).replace("\\", "/")] = sha(p)
    for relative, pairs in replacements.items():
        p = source / relative
        original = p.read_text()
        text = original
        for old, new in pairs:
            if old not in text:
                raise RuntimeError("Expected legacy syntax missing: " + old)
            text = text.replace(old, new)
        p.write_text(text)
        changes.extend(
            difflib.unified_diff(
                original.splitlines(True),
                text.splitlines(True),
                fromfile="author/" + relative,
                tofile="work/" + relative,
            )
        )
    (ROOT / "outputs/python3.patch").write_text("".join(changes))
    (ROOT / "outputs/author_source.json").write_text(
        json.dumps({"commit": COMMIT, "files": hashes}, indent=2)
    )
    return source


def execute(source, script, args):
    os.chdir(source / "rgcn")
    sys.path.insert(0, str(source))
    path = source / "rgcn" / script
    sys.argv = [str(path)] + args
    return runpy.run_path(str(path), run_name="__main__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    options = parser.parse_args()
    (ROOT / "outputs").mkdir(exist_ok=True)
    os.environ["KERAS_HOME"] = str(ROOT / "work/keras")
    source = prepare()
    execute(source, "prepare_dataset.py", ["-d", "aifb"])
    ns = execute(
        source,
        "train.py",
        [
            "-d",
            "aifb",
            "--bases",
            "0",
            "--hidden",
            "16",
            "--l2norm",
            "0.",
            "--testing",
            "--epochs",
            str(options.epochs),
        ],
    )
    import numpy as np

    predictions = ns["preds"]
    assert np.isfinite(predictions).all()
    np.save(ROOT / "outputs/predictions.npy", predictions)
    weights = ns["model"].get_weights()
    np.savez(ROOT / "outputs/weights.npz", *weights)
    report = {
        "status": "passed",
        "author_commit": COMMIT,
        "data": "Original public AIFB graph and author split",
        "epochs": options.epochs,
        "nodes": ns["num_nodes"],
        "relations_including_reverse_and_self": ns["support"],
        "train_people": len(ns["idx_train"]),
        "test_people": len(ns["idx_test"]),
        "test_loss": float(ns["test_loss"][0]),
        "test_accuracy": float(ns["test_acc"][0]),
        "prediction_shape": list(predictions.shape),
        "compatibility_patch": "Python 2 dictionary iteration, generator next, list range, string type and integer division adapted to Python 3.",
    }
    (ROOT / "outputs/result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
