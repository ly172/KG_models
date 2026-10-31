# Download and index the RxNorm Prescribable subset.

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_10052026.zip"
SHA256 = "8ed351c29f3cb56bedeb8e2bcb8fa56931f8d7eb88e2280dc9772d7d3eb16977"
MD5 = "a3bfab544c6a67c127ebcd10ab914aa9"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def build(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "RxNorm_full_prescribe_10052026.zip"
    if not archive.exists() or digest(archive) != SHA256:
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=180) as response, temporary.open(
            "wb"
        ) as stream:
            while block := response.read(1048576):
                stream.write(block)
        assert digest(temporary) == SHA256
        temporary.replace(archive)
    assert digest(archive, "md5") == MD5
    directory = root / "rxnorm_prescribable_20261005"
    directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith("rrf/") or name.startswith("Readme_"):
                target = directory / Path(name).name
                payload = z.read(name)
                if not target.exists() or target.read_bytes() != payload:
                    target.write_bytes(payload)
    ndc_map = defaultdict(set)
    ndc_all_sources = defaultdict(set)
    ndc_sources = Counter()
    counts = Counter()
    for line in (directory / "RXNSAT.RRF").open(encoding="utf-8"):
        counts["RXNSAT_rows"] += 1
        row = line.rstrip("\n").split("|")
        if row[8] == "NDC" and row[11] == "N":
            ndc_sources[row[9]] += 1
            ndc_all_sources[row[10]].add(row[0])
            if row[9] == "RXNORM":
                if re.fullmatch(r"[0-9]{11}", row[10]) is None:
                    raise ValueError(
                        "Official RXNORM NDC is not an exact 11-digit string."
                    )
                ndc_map[row[10]].add(row[0])
    names = defaultdict(list)
    tty_counts = Counter()
    source_counts = Counter()
    for line in (directory / "RXNCONSO.RRF").open(encoding="utf-8"):
        counts["RXNCONSO_rows"] += 1
        row = line.rstrip("\n").split("|")
        source_counts[row[11]] += 1
        if row[11] == "RXNORM" and row[16] == "N" and row[1] == "ENG":
            names[row[0]].append({"name": row[14], "term_type": row[12]})
            tty_counts[row[12]] += 1
    relation_counts = Counter()
    concept_relations = 0
    for line in (directory / "RXNREL.RRF").open(encoding="utf-8"):
        counts["RXNREL_rows"] += 1
        row = line.rstrip("\n").split("|")
        if row[0] and row[4] and row[10] == "RXNORM":
            concept_relations += 1
            relation_counts[row[7] or row[3]] += 1
    derived = {
        "ndc_to_rxcui.json": {
            key: sorted(values) for key, values in sorted(ndc_map.items())
        },
        "ndc_to_rxcui_all_public_sources.json": {
            key: sorted(values) for key, values in sorted(ndc_all_sources.items())
        },
        "rxcui_names.json": dict(sorted(names.items())),
    }
    for name, data in derived.items():
        (directory / name).write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    manifest = {
        "release": "2026-10-05",
        "download_url": URL,
        "source_page": "https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html",
        "documentation": "https://www.nlm.nih.gov/research/umls/rxnorm/docs/prescribe.html",
        "field_documentation": "https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html",
        "license": "NLM Current Prescribable Content; no license required; public domain.",
        "archive": {
            "file": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": SHA256,
            "md5": MD5,
        },
        "counts": dict(counts),
        "source_counts": dict(source_counts),
        "ndc_sources": dict(ndc_sources),
        "rxnorm_concepts_with_names": len(names),
        "rxnorm_ndc_codes": len(ndc_map),
        "rxnorm_ndc_ambiguous_codes": sum(len(v) > 1 for v in ndc_map.values()),
        "all_public_source_ndc_codes": len(ndc_all_sources),
        "ndc_length_counts": dict(Counter(len(k) for k in ndc_map)),
        "term_types": dict(tty_counts),
        "rxnorm_concept_relation_rows": concept_relations,
        "relation_types": dict(relation_counts),
        "lookup_rule": "ndc_to_rxcui uses exact ATV for ATN=NDC, SAB=RXNORM, SUPPRESS=N. All multiple targets are retained; no nearest-name matching or forced selection.",
        "relation_direction": "For RXNREL, the SECOND concept at zero-based field 4 has relation RELA (field 7) to the FIRST concept at field 0, per NLM documentation.",
        "semantic_types": "No RXNSTY or UMLS MRSTY in this subset. RxNorm term types such as SCD/IN are not UMLS semantic type IDs.",
        "known_limits": [
            "Only currently prescribable content; historical NDC values may be absent.",
            "RxCUI identifiers are not UMLS CUI identifiers; use a separately documented mapping.",
            "No ICD diagnosis/procedure vocabulary or original complete UMLS semantic-type table.",
        ],
        "files": [
            {"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(directory.iterdir())
            if p.is_file()
        ],
    }
    (root / "rxnorm_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        key: manifest[key]
        for key in [
            "counts",
            "rxnorm_concepts_with_names",
            "rxnorm_ndc_codes",
            "rxnorm_ndc_ambiguous_codes",
            "all_public_source_ndc_codes",
            "ndc_length_counts",
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
