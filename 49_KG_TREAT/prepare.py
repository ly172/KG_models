# Fix imports in a copy of the author's files.

from pathlib import Path
import ast, hashlib, json, shutil, difflib

ROOT = Path(__file__).resolve().parent
PATCHES = {
    "model/bert_custom.py": [
        (
            "from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward, find_pruneable_heads_and_indices, \\\n    prune_linear_layer",
            "from transformers.modeling_utils import PreTrainedModel\nfrom transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer",
        ),
        (
            "from transformers.modeling_utils import ModelOutput",
            "from transformers.utils import ModelOutput",
        ),
    ],
    "model/modeling_kgtreat.py": [
        (
            "from transformers.utils.versions import importlib_metadata",
            "from importlib import metadata as importlib_metadata",
        ),
        (
            "from transformers.modeling_utils import (\n    ModelOutput,\n    logging,\n)\nfrom transformers.file_utils import (\n    is_scatter_available\n)",
            'from transformers.utils import ModelOutput, logging\ndef is_scatter_available():\n    return importlib.util.find_spec("torch_scatter") is not None',
        ),
    ],
    "utils/configuration_utils.py": [
        (
            "from transformers.modeling_utils import logging, PretrainedConfig",
            "from transformers.utils import logging\nfrom transformers import PretrainedConfig",
        )
    ],
}


def prepare():
    manifest = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert (
            hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        ), entry["path"]
    dest = ROOT / "run/compatible"
    dest.mkdir(parents=True, exist_ok=True)
    diffs = []
    for entry in manifest["files"]:
        rel = entry["path"].removeprefix("upstream/")
        src = ROOT / entry["path"]
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel not in PATCHES:
            shutil.copyfile(src, target)
            continue
        before = src.read_text(encoding="utf-8")
        after = before
        for old, new in PATCHES[rel]:
            assert after.count(old) == 1, (rel, old)
            after = after.replace(old, new)
        target.write_text(after, encoding="utf-8")
        diffs.extend(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="upstream/" + rel,
                tofile="compatible/" + rel,
            )
        )
        original_classes = {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(before).body
            if isinstance(n, ast.ClassDef)
        }
        patched_classes = {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(after).body
            if isinstance(n, ast.ClassDef)
        }
        assert original_classes == patched_classes, ("class body changed", rel)
    (ROOT / "run/import_compatibility.patch").write_text(
        "".join(diffs), encoding="utf-8"
    )
    return dest


if __name__ == "__main__":
    prepare()
