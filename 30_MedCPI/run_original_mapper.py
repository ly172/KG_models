# Run the author's SapBERT mapper and compare it with public sources.

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_author_stages import environment
from prepare_public_inputs import lines, write_jsonl

ROOT = Path(__file__).resolve().parent


def clear_candidate_cache(mapping, kind):
    # The author's cache has no fingerprint, so re-encode on every run.
    if kind not in ["diagnosis", "procedure", "medication"]:
        raise ValueError("Unexpected mapping kind")
    cache = mapping / f"native_mapper/{kind}/{kind}_candidate_embeddings.pt"
    cache.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sapbert-root", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--worker", choices=["diagnosis", "procedure", "medication"])
    a = ap.parse_args()
    source_metadata = json.loads(a.source_manifest.read_text(encoding="utf-8"))
    manifest = source_metadata.get("sapbert", source_metadata)
    for item in manifest["files"]:
        if (
            hashlib.sha256((a.sapbert_root / item["path"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("Pretrained encoder checksum mismatch")
    mapping = ROOT / ".private/outputs/mapping"
    if a.worker:
        clear_candidate_cache(mapping, a.worker)
        os.environ.update(environment())
        sys.path.insert(0, str(ROOT / "author/scripts"))
        import sapbert_map_mentions as original

        original.MODEL_NAME = str(a.sapbert_root.resolve())
        sys.argv = [
            "sapbert_map_mentions.py",
            "--mentions",
            str(mapping / f"public_mentions/{a.worker}.jsonl"),
            "--candidates",
            str(mapping / f"public_candidates/{a.worker}.jsonl"),
            "--output-dir",
            str(mapping / f"native_mapper/{a.worker}"),
            "--source-type",
            a.worker,
            "--batch-size",
            "32",
            "--device",
            "cpu",
        ]
        original.main()
        return
    (ROOT / "verification/sapbert_source.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    expected = json.loads(
        (mapping / "expected_exact_ids.json").read_text(encoding="utf-8")
    )
    summary = {}
    for kind in ["diagnosis", "procedure", "medication"]:
        start = time.time()
        log = ROOT / ".private/logs" / f"sapbert_{kind}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--sapbert-root",
                    str(a.sapbert_root),
                    "--source-manifest",
                    str(a.source_manifest),
                    "--worker",
                    kind,
                ],
                env=environment(),
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if p.returncode:
            raise RuntimeError(f"Original {kind} mapper failed; private log retained")
        native = list(
            lines(mapping / f"native_mapper/{kind}/{kind}_mention_to_umls.jsonl")
        )
        result = []
        counts = Counter()
        rejected = []
        for row in native:
            counts["native_outputs"] += 1
            if row.get("matched_cui") == expected[row["mention_id"]]:
                counts["source_confirmed"] += 1
                counts["confirmed_" + row["retrieval_mode"]] += 1
                result.append(row)
            else:
                rejected.append(row)
                counts["native_output_disagrees_with_exact_source"] += 1
                result.append(row)
        missing = list(lines(mapping / f"public_unmapped/{kind}.jsonl"))
        counts["source_unavailable_or_ambiguous"] = len(missing)
        result.extend(missing)
        write_jsonl(
            mapping / f"sapbert/mimic4/{kind}/{kind}_mention_to_umls.jsonl", result
        )
        write_jsonl(
            mapping / f"native_mapper/{kind}/source_disagreements.jsonl", rejected
        )
        written = {
            row["mention_id"]: row
            for row in lines(
                mapping / f"sapbert/mimic4/{kind}/{kind}_mention_to_umls.jsonl"
            )
        }
        unchanged = all(written[row["mention_id"]] == row for row in native)
        if not unchanged:
            raise AssertionError("Author mapper output changed before alignment")
        summary[kind] = {
            "counts": dict(counts),
            "seconds": round(time.time() - start, 3),
            "original_mapper_exit_code": p.returncode,
            "all_native_outputs_passed_unchanged": unchanged,
        }
        print(json.dumps({kind: summary[kind]}), flush=True)
    summary["mapping_policy"] = (
        "All native author mapper outputs are passed to the author alignment and graph stages unchanged. Source disagreements are recorded separately."
    )
    summary["native_outputs_replaced_by_wrapper"] = 0
    (ROOT / "verification/mapping.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
