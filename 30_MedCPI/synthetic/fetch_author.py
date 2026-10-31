# Download the author's source files.

from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def fetch_author():
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    target = ROOT / "author"
    target.mkdir(exist_ok=True)
    for item in manifest["files"]:
        path = target / item["path"]
        if (
            path.exists()
            and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        ):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    item["url"], headers={"User-Agent": "kg_model-reproduction"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = response.read()
                if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError("Source checksum mismatch: " + item["path"])
                path.write_bytes(payload)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        print("verified author source: " + item["path"], flush=True)
    return target
