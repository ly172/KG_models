# Copy the author's source without unused imports.

import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definition_hash(text, name):
    tree = ast.parse(text)
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def prepare(repair_imports=True, output=None):
    output = Path(output or ROOT / ".run/author")
    manifest = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    removals = {
        "graphcare_/model.py": ["from pyhealth.models import RETAINLayer\n"],
        "graphcare_/__init__.py": ["from .splitter import *"],
        "graphcare.py": [
            "from pyhealth.datasets import SampleDataset\n",
            "from graphcare_ import split_by_patient\n",
            "from pyhealth.metrics import multilabel_metrics_fn\n",
            "import neptune\n",
        ],
    }
    definitions = {
        "graphcare_/model.py": ["GraphCare", "BiAttentionGNNConv"],
        "graphcare.py": [
            "train",
            "evaluate",
            "get_mode_and_out_channels_and_loss_func",
        ],
    }
    checks = {}
    patches = []
    for rel, digest in manifest["files"].items():
        source = ROOT / "upstream" / rel
        if sha256(source) != digest:
            raise RuntimeError("Author hash mismatch: " + rel)
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".py"):
            before = source.read_text(encoding="utf-8")
            after = before
            for old in removals.get(rel, []) if repair_imports else []:
                if after.count(old) != 1:
                    raise RuntimeError("Expected one unused import: " + old)
                after = after.replace(old, "")
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(after)
            for name in definitions.get(rel, []):
                original_hash = definition_hash(before, name)
                runtime_hash = definition_hash(after, name)
                assert original_hash == runtime_hash
                checks[rel + ":" + name] = {
                    "original": original_hash,
                    "runtime": runtime_hash,
                }
            patches.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="upstream/" + rel,
                    tofile="runtime/" + rel,
                )
            )
        else:
            shutil.copyfile(source, target)
    (output / "compatibility.patch").write_text("".join(patches), encoding="utf-8")
    (output / "core_source_check.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    return output, checks
