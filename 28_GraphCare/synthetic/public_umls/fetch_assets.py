# Download the public GraphCare resources to a shared cache.

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPOSITORY = next(
    parent
    for parent in (ROOT, *ROOT.parents)
    if (parent / "models.json").is_file() and (parent / "shared/downloads.py").is_file()
)
sys.path.insert(0, str(REPOSITORY / "shared"))
from downloads import download_file, sha256


def default_asset_directory():
    manifest = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    version = manifest["encoder_revision"] + "_" + manifest["author_graph_commit"]
    return REPOSITORY / ".run" / "resources" / "graphcare" / version


def ensure_assets(directory=None):
    directory = (
        Path(directory).resolve()
        if directory is not None
        else default_asset_directory()
    )
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    checked = []
    for item in manifest["assets"]:
        download_file(
            item["url"], directory / item["path"], item["sha256"], item["bytes"]
        )
        checked.append(item["path"])
    return directory, {
        "verified_files": checked,
        "total_bytes": sum(item["bytes"] for item in manifest["assets"]),
        "encoder_revision": manifest["encoder_revision"],
        "graph_commit": manifest["author_graph_commit"],
    }


if __name__ == "__main__":
    _, report = ensure_assets()
    print(json.dumps(report, indent=2))
