# Download the author's MedCPI source and schemas.

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
import sys

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

COMMIT = "9413431d8741dfb875d12a2a8d516c94643d9748"
SCRIPTS = [
    "preprocess_ehr_to_jsonl",
    "prepare_mapping_inputs",
    "build_umls_candidate_index",
    "sapbert_map_mentions",
    "build_aligned_patient_ehr",
    "build_task_instances",
    "create_task_splits",
    "export_global_relation_inventory",
    "induce_relation_schema",
    "extract_task_train_concepts",
    "export_relation_inventory",
    "build_concept_mkg",
    "verify_construct_outputs",
    "run_construct_pipeline",
    "build_patient_pkg",
    "pkg_utils",
    "prepare_integrate_inputs",
    "integrate_data",
    "integrate_dataset",
    "integrate_model",
    "train_integrate_formal",
    "path_roots",
]
FILES = (
    ["scripts/" + x + ".py" for x in SCRIPTS]
    + ["README.md", "requirements.txt"]
    + [
        "outputs/construct/shared/" + t + "/relation_schema.json"
        for t in ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]
    ]
)
PINNED = json.loads((ROOT / "author_source.json").read_text(encoding="utf-8"))
if PINNED["commit"] != COMMIT:
    raise ValueError("Author commit differs from the pinned manifest")
EXPECTED = {r["path"]: r for r in PINNED["files"]}
if set(EXPECTED) != set(FILES):
    raise ValueError("Pinned author file inventory differs")


def verify(path, data):
    expected = EXPECTED[path]
    if (
        len(data) != expected["bytes"]
        or hashlib.sha256(data).hexdigest() != expected["sha256"]
    ):
        raise ValueError("Author source checksum mismatch: " + path)


def fetch(path):
    url = f"https://raw.githubusercontent.com/hw146/MedCPI/{COMMIT}/{path}"
    expected = EXPECTED[path]
    p = ROOT / "author" / path
    download_file(url, p, expected["sha256"], expected["bytes"])
    data = p.read_bytes()
    verify(path, data)
    return {
        "path": path,
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-local", action="store_true")
    args = ap.parse_args()
    if args.verify_local:
        for path in FILES:
            verify(path, (ROOT / "author" / path).read_bytes())
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, FILES))
    print(
        json.dumps(
            {
                "verified_author_files": len(FILES),
                "commit": COMMIT,
            }
        )
    )
