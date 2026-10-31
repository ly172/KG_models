# Download the files in public_sources.json and rebuild terminology indexes.
# --cache can point to an earlier output folder.

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(
    0,
    str(
        next(
            parent / "shared"
            for parent in HERE.parents
            if (parent / "shared/downloads.py").is_file()
        )
    ),
)
from downloads import download_file


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def valid(path, item):
    return (
        path.is_file()
        and path.stat().st_size == item["bytes"]
        and digest(path) == item["sha256"]
    )


def fetch(root, item, cache):
    target = root / item["path"]
    method = "verified_existing" if valid(target, item) else "downloaded_and_verified"
    candidate = cache / item["path"] if cache is not None else None
    if (
        method != "verified_existing"
        and candidate is not None
        and valid(candidate, item)
    ):
        method = "verified_cache_copy"
    download_file(item["url"], target, item["sha256"], item["bytes"], cache=candidate)
    return method


def build(output, cache=None):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = json.loads((HERE / "public_sources.json").read_text(encoding="utf-8"))[
        "sources"
    ]
    receipts = []
    for item in sources:
        method = fetch(root, item, Path(cache).resolve() if cache else None)
        receipts.append(dict(**item, method=method))
        print(method + ": " + item["path"], flush=True)
    steps = [
        [
            "extract_public_terminology.py",
            "--output",
            str(root / "graphcare_terminology"),
            "--graphcare-root",
            str(root / "graphcare"),
        ],
        [
            "prepare_scispacy_types.py",
            "--output",
            str(root / "scispacy_public_subset"),
            "--graphcare-root",
            str(root / "graphcare"),
        ],
        ["prepare_rxnorm.py", "--output", str(root)],
        ["prepare_historical_rxnorm.py", "--output", str(root)],
        ["build_historical_rxnorm_indices.py", "--output", str(root)],
    ]
    for script, *args in steps:
        subprocess.run([sys.executable, str(HERE / script), *args], check=True)
    names = [
        "graphcare_terminology/records.jsonl",
        "graphcare_terminology/cui_types.json",
        "scispacy_public_subset/concepts.jsonl",
        "scispacy_public_subset/cui_types.json",
    ]
    names += [
        f"rxnorm_prescribable_{version}/{name}.json"
        for version in ["20141201", "20191202", "20261005"]
        for name in ["ndc_to_rxcui", "rxcui_names"]
    ]
    result = dict(
        assets=receipts,
        outputs=[
            dict(
                path=name,
                bytes=(root / name).stat().st_size,
                sha256=digest(root / name),
            )
            for name in names
        ],
    )
    (root / "public_resource_receipt.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    a = p.parse_args()
    build(a.output, a.cache)
