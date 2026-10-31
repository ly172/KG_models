# Download the author's code and the language model files.

from pathlib import Path
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    import truststore

    for name, source in json.loads((ROOT / "sources.json").read_text()).items():
        target = ROOT / "cache" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) == source["sha256"]:
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        print("Downloading", name, flush=True)
        for attempt in range(4):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                request = urllib.request.Request(
                    source["url"],
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                )
                with urllib.request.urlopen(
                    request,
                    context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                    timeout=120,
                ) as response:
                    with temporary.open(
                        "ab" if response.status == 206 and offset else "wb"
                    ) as f:
                        shutil.copyfileobj(response, f)
                break
            except Exception:
                if os.name == "nt":
                    command = [
                        "curl.exe",
                        "--fail",
                        "--location",
                        "--retry",
                        "3",
                        "--continue-at",
                        "-",
                        "--output",
                        str(temporary),
                        source["url"],
                    ]
                    if subprocess.run(command).returncode == 0:
                        break
                if attempt == 3:
                    raise
                time.sleep(2)
        if sha(temporary) != source["sha256"]:
            raise RuntimeError("Source checksum differs: " + name)
        temporary.replace(target)
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    for name, expected in provenance["files"].items():
        if sha(ROOT / "data" / name) != expected:
            raise RuntimeError("Public example differs: " + name)
