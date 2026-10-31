# Get the scispaCy KB and keep the CUI types it lists.

import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / "scispacy_public_subset"
URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/kbs/2023-04-23/umls_2022_ab_cat0129.jsonl"
EXPECTED_BYTES = 658702592


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphcare-root", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.output
    AUTHOR = args.graphcare_root
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "assets/umls_2022_ab_cat0129.jsonl"
    if not path.is_file() or path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            "Run prepare_public_resources.py to obtain the pinned public KB."
        )
    wanted = set(
        (AUTHOR / "umls/concepts.txt").read_text(encoding="utf-8").splitlines()
    )
    for name in ["ICD9CM_to_UMLS.csv", "ATC_to_UMLS.csv", "RxNorm_to_UMLS.csv"]:
        with (AUTHOR / name).open(encoding="utf-8-sig") as stream:
            wanted.update(row["UMLS"] for row in csv.DictReader(stream))
    seen = set()
    types = {}
    scanned = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream, (ROOT / "concepts.jsonl").open(
        "w", encoding="utf-8"
    ) as out:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            scanned += 1
            cui = row["concept_id"]
            if cui in wanted:
                seen.add(cui)
                types[cui] = sorted(set(row["types"]))
                # Keep aliases as given; definitions are not needed.
                out.write(
                    json.dumps(
                        {
                            k: row[k]
                            for k in [
                                "concept_id",
                                "canonical_name",
                                "aliases",
                                "types",
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    if (
        digest.hexdigest()
        != "1e5d2e0257ca7874daf559f5bf433ec5955bb98b9ae32f716007bd6288d2eeeb"
    ):
        raise ValueError("Public KB SHA256 mismatch")
    (ROOT / "cui_types.json").write_text(
        json.dumps(types, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "missing_cuis.json").write_text(
        json.dumps(sorted(wanted - seen)), encoding="utf-8"
    )
    manifest = dict(
        url=URL,
        sha256=digest.hexdigest(),
        bytes=path.stat().st_size,
        source_documentation="https://github.com/allenai/scispacy/blob/main/scispacy/linking_utils.py",
        source_export_code="https://github.com/allenai/scispacy/blob/main/scripts/export_umls_json.py",
        source_version="2022AB subset published 2023-04-23",
        original_concepts_scanned=scanned,
        requested_public_cuis=len(wanted),
        matched_public_cuis=len(seen),
        missing_public_cuis=len(wanted - seen),
        requested_set="Author GraphCare graph and its public ICD9CM, ATC and RxNorm CUI mapping tables",
        scope="Public author-distributed subset; not a complete MRCONSO/MRSTY release; source-vocabulary restrictions remain applicable.",
        outputs={
            n: dict(
                bytes=(ROOT / n).stat().st_size,
                sha256=hashlib.sha256((ROOT / n).read_bytes()).hexdigest(),
            )
            for n in ["concepts.jsonl", "cui_types.json", "missing_cuis.json"]
        },
    )
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
