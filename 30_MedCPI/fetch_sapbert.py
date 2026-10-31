# Download the SapBERT release used by the author, or use a local copy.

import argparse
import hashlib
import json
from pathlib import Path
import shutil
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


def prepare(cache=None):
    manifest = json.loads(
        (ROOT / "verification/sapbert_source.json").read_text(encoding="utf-8")
    )
    destination = ROOT / ".assets/sapbert"
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        target = destination / item["path"]
        download_file(
            item["url"],
            target,
            item["sha256"],
            item.get("bytes"),
            cache=Path(cache) / item["path"] if cache else None,
        )
    print(
        json.dumps(
            {
                "sapbert_fixed_revision": manifest["revision"],
                "verified_files": len(manifest["files"]),
            }
        ),
        flush=True,
    )
    return destination


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path)
    args = ap.parse_args()
    prepare(args.cache)
