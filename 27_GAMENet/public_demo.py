# Map MIMIC-IV demo visits to the author's GAMENet vocabulary.

import ast
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import random
import sys

import dill
from read_legacy_vocabulary import convert

ROOT = Path(__file__).resolve().parent
sys.path.insert(
    0,
    str(
        next(
            parent / "shared"
            for parent in ROOT.parents
            if (parent / "shared/downloads.py").is_file()
        )
    ),
)
from downloads import download_file
from mimic_demo import ensure_demo


def generate(author):
    data = Path(author) / "data"
    vocabulary, conversion = convert(
        ROOT / "upstream/data/voc_final.pkl", data / "voc_final.pkl"
    )
    demo = ensure_demo()
    assets = ROOT / "public_tables"
    for item in json.loads((ROOT / "public_tables.json").read_text()):
        download_file(item["url"], assets / item["file"], item["sha256"], item["bytes"])
    ndc_to_rx = ast.literal_eval((assets / "ndc2rxnorm_mapping.txt").read_text())
    rx_to_atc = {}
    with (assets / "ndc2atc_level4.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rx_to_atc.setdefault(row["RXCUI"], row["ATC4"][:4])

    def read(name):
        with gzip.open(demo / "hosp" / name, "rt", newline="") as stream:
            yield from csv.DictReader(stream)

    indices = {
        key: {str(code): int(index) for code, index in vocabulary[key].word2idx.items()}
        for key in ("diag_voc", "pro_voc", "med_voc")
    }
    admissions = {row["hadm_id"]: row for row in read("admissions.csv.gz")}
    visits = defaultdict(lambda: [set(), set(), set()])
    counts = defaultdict(int)
    for name, key, position in [
        ("diagnoses_icd.csv.gz", "diag_voc", 0),
        ("procedures_icd.csv.gz", "pro_voc", 1),
    ]:
        for row in read(name):
            counts[name + "_rows"] += 1
            if row["icd_version"] != "9":
                continue
            code = row["icd_code"].replace(".", "").strip()
            if code in indices[key]:
                visits[row["hadm_id"]][position].add(indices[key][code])
                counts[name + "_matched"] += 1
    for row in read("prescriptions.csv.gz"):
        counts["prescription_rows"] += 1
        ndc = row["ndc"].strip().zfill(11)
        atc = rx_to_atc.get(str(ndc_to_rx.get(ndc, "")))
        if atc in indices["med_voc"]:
            visits[row["hadm_id"]][2].add(indices["med_voc"][atc])
            counts["prescription_rows_matched"] += 1
    by_patient = defaultdict(list)
    for hadm, parts in visits.items():
        if hadm in admissions and all(parts):
            row = admissions[hadm]
            by_patient[row["subject_id"]].append(
                (row["admittime"], hadm, [sorted(x) for x in parts])
            )
    patients = [
        (key, sorted(value))
        for key, value in sorted(by_patient.items())
        if len(value) >= 2
    ]
    random.Random(20261030).shuffle(patients)
    if len(patients) < 12:
        raise RuntimeError(
            "Fewer than twelve patients have two complete visits in the author vocabulary."
        )
    records = [[visit[2] for visit in visits] for _, visits in patients]
    with (data / "records_final.pkl").open("wb") as stream:
        dill.dump(records, stream)
    train_count = int(len(records) * 2 / 3)
    test_count = int((len(records) - train_count) / 2)
    report = dict(
        kind="public_MIMIC_IV_Demo_2_2",
        source="https://physionet.org/content/mimic-iv-demo/2.2/",
        patients=len(records),
        visits=sum(map(len, records)),
        matching_counts=dict(counts),
        vocab_sizes={key: len(value) for key, value in indices.items()},
        split=dict(
            train_patients=train_count,
            test_patients=test_count,
            validation_patients=len(records) - train_count - test_count,
        ),
        selection="Patients with at least two admissions containing matched ICD-9 diagnoses, ICD-9 procedures and medications.",
        mapping="Exact author ICD-9 vocabulary match; author NDC to RxNorm table, first author ATC entry per RxNorm code, and four-character ATC medication groups.",
        excluded="ICD-10 codes, unmatched prescriptions, visits missing a required code group, and patients with fewer than two retained visits.",
        graphs="Author public EHR medication co-occurrence and DDI graphs.",
        vocabulary_serialization=conversion,
    )
    (data / "public_demo_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
