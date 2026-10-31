# Build NDC indexes for each RxNorm release.

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RELEASES = [("2014-12-01", 179807), ("2019-12-02", 234400)]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    ROOT = parser.parse_args().output.resolve()
    versions = {}
    manifests = []
    receipts = {
        r["release"]: r
        for r in json.loads(
            (ROOT / "rxnorm_historical_downloads.json").read_text(encoding="utf-8")
        )["releases"]
    }
    for release, official_count in RELEASES:
        folder = ROOT / ("rxnorm_prescribable_" + release.replace("-", ""))
        mapping = defaultdict(set)
        count = Counter()
        for line in (folder / "RXNSAT.RRF").open(encoding="utf-8"):
            row = line.rstrip("\n").split("|")
            count["RXNSAT_rows"] += 1
            if row[8] == "NDC" and row[9] == "RXNORM" and row[11] == "N":
                if re.fullmatch(r"[0-9]{11}", row[10]) is None:
                    raise ValueError(
                        "Official RXNORM NDC is not an exact 11-digit string: "
                        + release
                    )
                mapping[row[10]].add(row[0])
                count["RXNORM_NDC_rows"] += 1
        assert len(mapping) == official_count, (release, len(mapping), official_count)
        normalized = {ndc: sorted(values) for ndc, values in sorted(mapping.items())}
        save(folder / "ndc_to_rxcui.json", normalized)
        names = defaultdict(list)
        for line in (folder / "RXNCONSO.RRF").open(encoding="utf-8"):
            row = line.rstrip("\n").split("|")
            count["RXNCONSO_rows"] += 1
            if row[1] == "ENG" and row[11] == "RXNORM" and row[16] == "N":
                names[row[0]].append({"name": row[14], "term_type": row[12]})
        save(folder / "rxcui_names.json", dict(sorted(names.items())))
        manifest = dict(
            receipts[release],
            selection="December monthly release of the requested calendar year, chosen before examining patient coverage or model outputs.",
            official_distinct_rxnorm_ndc_count=official_count,
            parsed_ndc_count=len(mapping),
            count=dict(count),
            within_release_ambiguous_ndcs=sum(len(v) > 1 for v in mapping.values()),
            ndc_length_counts=dict(Counter(len(k) for k in mapping)),
            rxcui_names=len(names),
            rule="ATN=NDC, SAB=RXNORM, SUPPRESS=N; exact stored NDC string; preserve all target RxCUIs.",
            release_notes="https://www.nlm.nih.gov/research/umls/rxnorm/docs/"
            + release[:4]
            + "/rxnorm_releasenotes_prescribe_"
            + release[5:7]
            + release[8:10]
            + release[:4]
            + ".html",
            files=[
                dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(folder.iterdir())
                if p.is_file() and p.name != "manifest.json"
            ],
        )
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        manifests.append(manifest)
        versions[release] = normalized
    versions["2026-10-05"] = json.loads(
        (ROOT / "rxnorm_prescribable_20261005/ndc_to_rxcui.json").read_text(
            encoding="utf-8"
        )
    )
    versioned = {}
    for version, mapping in versions.items():
        for ndc, targets in mapping.items():
            versioned.setdefault(ndc, {})[version] = targets
    versioned = dict(sorted(versioned.items()))
    conflicts = {
        ndc: version_targets
        for ndc, version_targets in versioned.items()
        if len({target for targets in version_targets.values() for target in targets})
        > 1
    }
    absent_current = {
        ndc: version_targets
        for ndc, version_targets in versioned.items()
        if "2026-10-05" not in version_targets
    }
    consistent_additions = {
        ndc: dict(
            rxcui=next(
                iter(
                    {
                        target
                        for targets in version_targets.values()
                        for target in targets
                    }
                )
            ),
            sources=version_targets,
        )
        for ndc, version_targets in absent_current.items()
        if ndc not in conflicts
    }
    out = ROOT / "rxnorm_historical_comparison"
    out.mkdir(exist_ok=True)
    save(out / "ndc_versioned_candidates.json", versioned)
    save(out / "ndc_cross_version_conflicts.json", conflicts)
    save(out / "ndc_missing_current_with_historical_sources.json", absent_current)
    save(out / "consistent_historical_additions.json", consistent_additions)
    summary = dict(
        releases=list(versions),
        union_ndcs=len(versioned),
        different_rxcuids_across_versions=len(conflicts),
        absent_from_current=len(absent_current),
        consistent_single_target_historical_additions=len(consistent_additions),
        ambiguous_historical_additions=len(absent_current) - len(consistent_additions),
        conflict_meaning="The exact target RxCUI identifiers differ. Some differences may be official retirements or merges; no equivalence is assumed here.",
        selection_policy="Keep each release separately. For current-missing NDCs only, consistent_historical_additions exposes unambiguous historical targets with all release provenance. Cross-version conflicts remain unresolved. No version is chosen from model performance.",
        historical_releases=manifests,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "historical_releases"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
