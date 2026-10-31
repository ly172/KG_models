# Make a patched copy of the bundled author code.

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    manifest = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    target = ROOT / ".run" / "source"
    changes = []
    patches = []
    for relative, expected in manifest["files"].items():
        original = ROOT / "upstream" / relative
        if sha256(original) != expected:
            raise RuntimeError(f"Author source hash mismatch: {relative}")
        before = original.read_text(encoding="utf-8")
        after = before
        rules = []
        if relative in ("train.py", "eval.py"):
            rules += [
                (
                    "num_workers=4",
                    "num_workers=0",
                    "Avoid Windows process spawning in the unguarded author entry point.",
                )
            ]
            loader = "train_loader" if relative == "train.py" else "eval_loader"
            rules += [
                (
                    f"{loader}.__iter__().next()",
                    f"next(iter({loader}))",
                    "Use the current Python iterator interface.",
                )
            ]
        if relative == "utils.py":
            rules += [
                (
                    "return zeros.scatter_(1, indices.unsqueeze(1), 1)",
                    "return zeros.scatter_(1, indices.long().unsqueeze(1), 1)",
                    "Windows native action integers can be int32; scatter requires int64 indices. The integer action values are preserved.",
                )
            ]
        if relative == "envs/block_pushing.py":
            rules += [
                (
                    "skimage.draw.circle(\n                        pos[0]*10 + 5, pos[1]*10 + 5, 5, im.shape)",
                    "skimage.draw.disk(\n                        (pos[0]*10 + 5, pos[1]*10 + 5), 5, shape=im.shape[:2])",
                    "Use the renamed filled-circle drawing function in Shapes.",
                ),
                (
                    "return state_obs, reward, done, None",
                    "return state_obs, reward, done, {}",
                    "Gym TimeLimit stores its episode-end marker in the info dictionary.",
                ),
            ]
        for old, new, reason in rules:
            if after.count(old) != 1:
                raise RuntimeError(
                    f"Patch must match exactly once: {relative}: {old!r}"
                )
            after = after.replace(old, new)
            changes.append(
                {"file": relative, "before": old, "after": new, "reason": reason}
            )
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(after, encoding="utf-8", newline="\n")
        if before != after:
            patches.extend(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="upstream/" + relative,
                    tofile="compatible/" + relative,
                )
            )
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "applied.patch").write_text("".join(patches), encoding="utf-8")
    (output / "changes.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


if __name__ == "__main__":
    print(prepare())
