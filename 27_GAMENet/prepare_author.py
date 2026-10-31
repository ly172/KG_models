# Copy the author's code and adjust imports, device and epochs.

import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output=None, repair_imports=True, cpu=True):
    output = Path(output or ROOT / ".run/author")
    manifest = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    edits = {
        "code/models.py": [("from dnc import DNC\n", "")] if repair_imports else [],
        "code/util.py": [("jaccard_similarity_score, ", "")] if repair_imports else [],
        "code/train_GAMENet.py": (
            [
                (
                    "device = torch.device('cuda:0')",
                    "device = torch.device(os.environ.get('KG_DEVICE', 'cuda:0'))",
                ),
                ("EPOCH = 40", "EPOCH = int(os.environ.get('KG_EPOCHS', '40'))"),
            ]
            if cpu
            else []
        ),
    }
    diff = []
    for rel, digest in manifest["files"].items():
        source = ROOT / "upstream" / rel
        if sha256(source) != digest:
            raise RuntimeError("Author hash mismatch: " + rel)
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".py"):
            before = source.read_text(encoding="utf-8")
            after = before
            for old, new in edits.get(rel, []):
                if after.count(old) != 1:
                    raise RuntimeError("Expected exactly one patch location: " + old)
                after = after.replace(old, new)
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(after)
            diff.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="upstream/" + rel,
                    tofile="runtime/" + rel,
                )
            )
        else:
            shutil.copyfile(source, target)
    (output / "compatibility.patch").write_text("".join(diff), encoding="utf-8")
    return output
