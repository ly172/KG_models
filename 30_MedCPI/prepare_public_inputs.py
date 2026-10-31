# Convert public terminology files to the author's input formats.

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def lines(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def iter_csv_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources-root", type=Path, required=True)
    ap.add_argument("--graphcare-root", type=Path, required=True)
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--semgroups", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "author/scripts"))
    from prepare_mapping_inputs import extract_mentions_for_dataset
    from build_umls_candidate_index import DIAG_SEMTYPES, PROC_SEMTYPES, MEDICATION_TTYS
    from sapbert_map_mentions import normalize_text as author_normalize_text

    # Same normalization as the author's mapper, which checks it again.
    def normalize(text):
        return " ".join(text.lower().strip().split())

    assets = ROOT / ".assets"
    umls = assets / "umls"
    umls.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.semgroups, umls / "SemGroups.txt")
    semantic = {}
    for line in a.semgroups.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            semantic[parts[2]] = parts[3]
    types = json.loads(
        (a.resources_root / "scispacy_public_subset/cui_types.json").read_text(
            encoding="utf-8"
        )
    )
    ttltypes = json.loads(
        (a.resources_root / "graphcare_terminology/cui_types.json").read_text(
            encoding="utf-8"
        )
    )
    for cui, value in ttltypes.items():
        if cui not in types:
            types[cui] = value
    names = {}
    with (a.graphcare_root / "umls/concept_names.txt").open(encoding="utf-8") as f:
        for line in f:
            cui, name = line.rstrip("\n").split("\t", 1)
            names[cui] = name
    for r in lines(a.resources_root / "graphcare_terminology/records.jsonl"):
        for cui in r["cuis"]:
            if cui not in names and r["names"]:
                names[cui] = r["names"][0]
    # Fields missing from the public files stay empty.
    with (umls / "MRCONSO.RRF").open("w", encoding="utf-8") as f:
        for cui, name in sorted(names.items()):
            row = [""] * 18
            row[0] = cui
            row[1] = "ENG"
            row[14] = name
            f.write("|".join(row) + "|\n")
    with (umls / "MRSTY.RRF").open("w", encoding="utf-8") as f:
        for cui, tuis in sorted(types.items()):
            for tui in tuis:
                if tui in semantic:
                    f.write("|".join([cui, tui, "", semantic[tui], "", ""]) + "|\n")
    relations = Counter()
    with (a.graphcare_root / "umls/umls.csv").open(encoding="utf-8") as source, (
        umls / "MRREL.RRF"
    ).open("w", encoding="utf-8") as target:
        for line in source:
            rel, src, dst, weight = line.rstrip("\n").split("\t")
            row = [""] * 16
            row[0] = src
            row[4] = dst
            row[7] = rel
            target.write("|".join(row) + "|\n")
            relations[rel] += 1
    codebook = {
        "icd9_diagnosis": {},
        "icd9_procedure": {},
        "icd10_diagnosis": {},
        "icd10_procedure": {},
    }
    from prepare_cohort import resolve_table

    for kind, table in [
        ("diagnosis", "d_icd_diagnoses"),
        ("procedure", "d_icd_procedures"),
    ]:
        source_path = resolve_table(a.mimic_root, table)
        for row in iter_csv_rows(source_path):
            version = row["icd_version"]
            code = row["icd_code"]
            title = row["long_title"]
            codebook[f"icd{version}_{kind}"][code] = {
                "long": title,
                "short": title,
                "long_title": title,
            }
    mapping = ROOT / ".private/outputs/mapping"
    extract_mentions_for_dataset(
        "mimic4",
        ROOT / ".private/outputs/preprocessed/mimic4/patients_visits.jsonl",
        codebook,
        mapping,
    )
    mentions = list(lines(mapping / "mentions/mimic4_unique_mentions.jsonl"))
    by_code = defaultdict(list)
    for r in lines(a.resources_root / "graphcare_terminology/records.jsonl"):
        if r["source"] == "ICD9CM":
            by_code[r["code"]].append(r)
    rxmap = defaultdict(set)
    with (a.graphcare_root / "RxNorm_to_UMLS.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        for r in csv.DictReader(f):
            rxmap[r["RxNorm"]].add(r["UMLS"])
    versions = ["20261005", "20191202", "20141201"]
    ndcs = {
        v: json.loads(
            (a.resources_root / f"rxnorm_prescribable_{v}/ndc_to_rxcui.json").read_text(
                encoding="utf-8"
            )
        )
        for v in versions
    }
    versioned_names = {
        v: json.loads(
            (a.resources_root / f"rxnorm_prescribable_{v}/rxcui_names.json").read_text(
                encoding="utf-8"
            )
        )
        for v in versions
    }
    rxnames = defaultdict(list)
    name_versions = defaultdict(set)
    for version, rows in versioned_names.items():
        for rx, items in rows.items():
            for item in items:
                if item not in rxnames[rx]:
                    rxnames[rx].append(item)
                name_versions[(rx, item["name"], item["term_type"])].add(version)
    name_rx = defaultdict(set)
    for rx, items in rxnames.items():
        for item in items:
            if item["term_type"] in MEDICATION_TTYS:
                name_rx[normalize(item["name"])].add(rx)
    candidates = {kind: {} for kind in ["diagnosis", "procedure", "medication"]}
    selected = defaultdict(list)
    unmapped = defaultdict(list)
    counts = Counter()
    expected = {}
    for mention in mentions:
        kind = mention["source_type"]
        possible = []
        reason = "no_exact_public_mapping"
        if kind in ["diagnosis", "procedure"]:
            code = mention["normalized_code"]
            split = 4 if code.startswith("E") else 3
            if kind == "procedure":
                split = 2
            dotted = code[:split] + ("." + code[split:] if len(code) > split else "")
            for record in by_code.get(dotted, []):
                for cui in record["cuis"]:
                    stys = [semantic[t] for t in types.get(cui, []) if t in semantic]
                    allowed = DIAG_SEMTYPES if kind == "diagnosis" else PROC_SEMTYPES
                    if not set(stys) & allowed:
                        continue
                    for term in record["names"]:
                        possible.append(
                            dict(
                                cui=cui,
                                umls_cui=cui,
                                sab="ICD9CM",
                                tty="",
                                code=dotted,
                                term_text=term,
                                semantic_types=stys,
                                source_type=kind,
                            )
                        )
            if len({r["cui"] for r in possible}) != 1:
                possible = []
                reason = "absent_or_ambiguous_code_or_semantic_type"
        else:
            ndc_code = mention.get("ndc") or ""
            versions_found = {
                v: ndcs[v][ndc_code]
                for v in versions
                if len(ndc_code) == 11 and ndc_code in ndcs[v]
            }
            ndc_options = {rx for vals in versions_found.values() for rx in vals}
            options = ndc_options if len(ndc_options) == 1 else set()
            mode = "exact_ndc"
            if len(ndc_options) > 1:
                counts["medication_ndc_cross_version_conflict"] += 1
            if ndc_code and len(ndc_code) != 11:
                counts["medication_ndc_non11_digits_unconverted"] += 1
            mapping_versions = list(versions_found)
            if not options:
                mode = "exact_name"
                for alias in mention.get("text_aliases", []):
                    options.update(name_rx.get(normalize(alias["text"]), set()))
                mapping_versions = versions
            if len(options) == 1:
                rx = next(iter(options))
                cu = rxmap.get(rx, set())
                cui = next(iter(cu)) if len(cu) == 1 else None
                for item in rxnames.get(rx, []):
                    if item["term_type"] not in MEDICATION_TTYS:
                        continue
                    possible.append(
                        dict(
                            cui=cui or "RXNORM:" + rx,
                            umls_cui=cui,
                            rxcui=rx,
                            sab="RXNORM",
                            tty=item["term_type"],
                            code=rx,
                            term_text=item["name"],
                            semantic_types=[
                                semantic[t] for t in types.get(cui, []) if t in semantic
                            ],
                            source_type=kind,
                            ndc_codes=[mention["ndc"]] if mode == "exact_ndc" else [],
                            public_mapping_versions=mapping_versions,
                            public_term_versions=sorted(
                                name_versions[(rx, item["name"], item["term_type"])]
                            ),
                        )
                    )
                counts["medication_" + mode] += 1
                if mode == "exact_ndc":
                    counts[
                        (
                            "medication_ndc_current"
                            if "20261005" in versions_found
                            else "medication_ndc_historical_only"
                        )
                    ] += 1
            else:
                reason = "absent_or_ambiguous_ndc_and_exact_name"
        if not mention.get("mention_text"):
            possible = []
            reason = "no_original_code_description"
        if not possible:
            unmapped[kind].append(
                dict(
                    **mention,
                    matched_cui=None,
                    matched_umls_cui=None,
                    retrieval_mode="public_resource_unavailable_or_ambiguous",
                    unmapped_reason=reason,
                )
            )
            counts[kind + "_unmapped"] += 1
            continue
        counts[kind + "_mapped"] += 1
        selected[kind].append(mention)
        expected[mention["mention_id"]] = possible[0]["cui"]
        for c in possible:
            key = (c["cui"], c["sab"], c["code"], c["term_text"])
            if key not in candidates[kind]:
                candidates[kind][key] = c
            elif c.get("ndc_codes"):
                candidates[kind][key]["ndc_codes"] = sorted(
                    set(candidates[kind][key].get("ndc_codes", []) + c["ndc_codes"])
                )
    for kind, rows in candidates.items():
        values = []
        for i, c in enumerate(rows.values()):
            values.append(
                dict(
                    candidate_id=f"{kind}_{i}",
                    normalized_text=author_normalize_text(c["term_text"]),
                    **c,
                )
            )
        write_jsonl(mapping / f"public_candidates/{kind}.jsonl", values)
        write_jsonl(mapping / f"public_mentions/{kind}.jsonl", selected[kind])
        write_jsonl(mapping / f"public_unmapped/{kind}.jsonl", unmapped[kind])
    (mapping / "expected_exact_ids.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    summary = {
        "semantic_type_source": "SciSpaCy official UMLS2022AB; GraphCare TTL only when the CUI is absent from SciSpaCy",
        "knowledge_direction": "Preserve GraphCare umls.csv first endpoint as CUI1 and second endpoint as CUI2; retain exact relation text. Source release REL/SAB/AUI/RUI/SUPPRESS not available and left blank.",
        "known_projection_limit": "Full original MRREL release direction, source vocabulary and suppression flags cannot be recovered from the public graph export.",
        "projected_concepts_with_names": len(names),
        "projected_semantic_profiles": len(types),
        "projected_relation_rows": sum(relations.values()),
        "candidate_counts": {k: len(v) for k, v in candidates.items()},
        "unique_mention_coverage": dict(counts),
        "mapping_policy": "Only a unique exact ICD code with original semantic-type eligibility, or unique exact official NDC/name RxCUI; unresolved inputs retained as unmapped. Original SapBERT mapper will run on these candidate sets.",
        "rxnorm_versions": versions,
        "ndc_policy": "Preserve source leading zeroes; only 11-digit exact NDC accepted. All available versions must agree on a single RxCUI. Current release preferred, historical consistent entries fill gaps. Conflicting NDC may be resolved only independently by a unique exact drug-name match. No 10-digit padding.",
    }
    (ROOT / "verification/public_input_projection.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
