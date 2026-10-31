# Convert the author's source to Python 3.

import difflib
import hashlib
import json
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(repair_empty=True):
    manifest = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    output = ROOT / ".run/author"
    output.mkdir(parents=True, exist_ok=True)
    patches = []
    for name, expected in manifest["files"].items():
        original = ROOT / "upstream" / name
        if sha256(original) != expected:
            raise RuntimeError("Author source hash mismatch: " + name)
        before = original.read_text(encoding="utf-8")
        after = (
            str(tool.refactor_string(before, name)) if name.endswith(".py") else before
        )
        if name == "gram.py" and repair_empty:
            old = "    treeMap = pickle.load(open(treeFile, 'rb'))\n"
            new = (
                old
                + "    if not treeMap:\n        width = int(treeFile.rsplit('.level', 1)[1].split('.')[0]) + 1\n        empty = np.empty((0, width), dtype='int32')\n        return empty, empty.copy()\n"
            )
            if after.count(old) != 1:
                raise RuntimeError("Expected one author tree loader")
            after = after.replace(old, new)
        with (output / name).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(after)
        if before != after:
            patches.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="upstream/" + name,
                    tofile="python3/" + name,
                )
            )
    (ROOT / ".run/python3_compatibility.patch").write_text(
        "".join(patches), encoding="utf-8"
    )
    return output
