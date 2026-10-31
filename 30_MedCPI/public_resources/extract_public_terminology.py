# Read code, CUI and type statements from the author's RDF files.

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
from rdflib import Graph, Namespace, RDF, OWL

COMMIT = "fe92aa67add80b62d4b5108d570b6483ee69a5d4"
BASE = f"https://raw.githubusercontent.com/pat-jj/GraphCare/{COMMIT}/KG_mapping/"
UMLS = Namespace("http://bioportal.bioontology.org/ontologies/umls/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphcare-root", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.output
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = []
    records = []
    all_types = defaultdict(set)
    for name in [
        "ICD9CM.ttl",
        "ATC.ttl",
        "ICD9CM_to_UMLS.csv",
        "ATC_to_UMLS.csv",
        "RxNorm_to_UMLS.csv",
    ]:
        payload = (args.graphcare_root / name).read_bytes()
        p = ROOT / "assets" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        sources.append(
            dict(
                path="assets/" + name,
                url=BASE + name,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    summaries = {}
    for ontology in ["ICD9CM", "ATC"]:
        graph = Graph().parse(ROOT / "assets" / f"{ontology}.ttl", format="turtle")
        source_pairs = set()
        count = 0
        for subject in sorted(graph.subjects(RDF.type, OWL.Class), key=str):
            if not str(subject).startswith(
                f"http://purl.bioontology.org/ontology/{ontology}/"
            ):
                continue
            codes = sorted(set(str(v) for v in graph.objects(subject, SKOS.notation)))
            cuis = sorted(set(str(v) for v in graph.objects(subject, UMLS.cui)))
            tuis = sorted(set(str(v) for v in graph.objects(subject, UMLS.tui)))
            hassty = sorted(
                set(
                    str(v).rsplit("/", 1)[-1]
                    for v in graph.objects(subject, UMLS.hasSTY)
                )
            )
            assert tuis == hassty, str(subject)
            labels = sorted(set(str(v) for v in graph.objects(subject, SKOS.prefLabel)))
            aliases = sorted(set(str(v) for v in graph.objects(subject, SKOS.altLabel)))
            assert codes and cuis and labels, str(subject)
            for code in codes:
                records.append(
                    dict(
                        source=ontology,
                        code=code,
                        cuis=cuis,
                        types=tuis,
                        names=labels,
                        aliases=aliases,
                        rdf_subject=str(subject),
                    )
                )
                for cui in cuis:
                    source_pairs.add((code, cui))
                    all_types[cui].update(tuis)
            count += 1
        with (ROOT / "assets" / f"{ontology}_to_UMLS.csv").open(
            encoding="utf-8-sig"
        ) as stream:
            published = {(row[ontology], row["UMLS"]) for row in csv.DictReader(stream)}
        summaries[ontology] = dict(
            rdf_classes=count,
            explicit_code_cui_pairs=len(source_pairs),
            published_csv_pairs=len(published),
            csv_pairs_missing_from_rdf=len(published - source_pairs),
            rdf_pairs_not_in_csv=len(source_pairs - published),
        )
        assert source_pairs == published, summaries[ontology]
    records.sort(key=lambda r: (r["source"], r["code"]))
    (ROOT / "records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    (ROOT / "cui_types.json").write_text(
        json.dumps({c: sorted(t) for c, t in sorted(all_types.items())}, indent=2),
        encoding="utf-8",
    )
    summary = dict(
        author_commit=COMMIT,
        source_ontology_version="2022ab",
        records=len(records),
        typed_cuis=len(all_types),
        ontology_checks=summaries,
        sources=sources,
        limitations=[
            "Public ICD9CM and ATC exports only; not complete MRCONSO or MRSTY.",
            "Codes, CUI links and semantic types come from explicit author RDF assertions.",
            "Multiple CUIs and types are retained; no forced one-to-one mapping.",
            "No patient records are read by this extractor.",
        ],
    )
    (ROOT / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "sources"}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
