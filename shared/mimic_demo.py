# Download the MIMIC-IV Demo 2.2.

import argparse
import json
from pathlib import Path
import shutil
import zipfile
from downloads import download_file, sha256

HERE = Path(__file__).resolve().parent
URL = "https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"
ARCHIVE_SHA256 = "97301a03820e8f41af211cf3462ddc19aefe75bbed05f11753859affaafeb8ec"
ARCHIVE_BYTES = 16189661
PREFIX = "mimic-iv-clinical-database-demo-2.2/"


def ensure_demo(output=None, cache=None):
    root = Path(output or HERE.parent / ".run/mimic_demo_2_2").resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "mimic_demo_files.json").read_text(encoding="utf-8"))
    archive = root / "mimic-iv-demo-2.2.zip"
    cache_file = None
    if cache:
        cache_path = Path(cache)
        cache_file = cache_path if cache_path.is_file() else cache_path / archive.name
    download_file(URL, archive, ARCHIVE_SHA256, ARCHIVE_BYTES, cache=cache_file)
    with zipfile.ZipFile(archive) as source:
        for item in manifest["files"]:
            target = (root / item["path"]).resolve()
            if root not in target.parents:
                raise ValueError("Unexpected archive destination")
            if target.exists() and sha256(target) == item["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with source.open(PREFIX + item["path"]) as stream, temporary.open(
                "wb"
            ) as out:
                shutil.copyfileobj(stream, out)
            if sha256(temporary) != item["sha256"]:
                raise ValueError("Official file checksum mismatch: " + item["path"])
            temporary.replace(target)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "name": "MIMIC-IV Clinical Database Demo",
                "version": "2.2",
                "public": True,
                "patients": 100,
                "url": "https://physionet.org/content/mimic-iv-demo/2.2/",
                "doi": "10.13026/dp1f-ex47",
                "license": "Open Data Commons Open Database License v1.0",
                "verified_files": len(manifest["files"]),
                "archive_sha256": ARCHIVE_SHA256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    print(ensure_demo(args.output, args.cache))
