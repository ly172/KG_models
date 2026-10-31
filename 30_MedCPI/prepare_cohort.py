# Select a fixed MIMIC-IV subset; patient files stay in .private.

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def resolve_table(root, name):
    for p in [
        root / "hosp" / f"{name}.csv" / f"{name}.csv",
        root / "hosp" / f"{name}.csv",
        root / "hosp" / f"{name}.csv.gz",
    ]:
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--patients", type=int, default=256)
    args = ap.parse_args()
    import duckdb

    private = ROOT / ".private"
    raw = private / "raw/mimic4"
    raw.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "verification"
    evidence.mkdir(exist_ok=True)
    db = duckdb.connect()
    db.execute("SET threads=2")
    tables = ["admissions", "diagnoses_icd", "procedures_icd", "prescriptions"]
    paths = {n: resolve_table(args.mimic_root, n) for n in tables}
    for n in tables:
        path = str(paths[n]).replace("'", "''")
        db.execute(
            f"CREATE VIEW {n} AS SELECT * FROM read_csv('{path}',all_varchar=true)"
        )
    # Cohort: 2-4 complete admissions, ICD-9 histories only.
    eligible = db.execute("""
        WITH a AS (SELECT subject_id,count(*) AS n,bool_and(admittime IS NOT NULL AND dischtime IS NOT NULL) AS timed FROM admissions GROUP BY subject_id),
        d AS (SELECT subject_id,bool_and(icd_version='9') AS only9,count(DISTINCT hadm_id) AS n FROM diagnoses_icd GROUP BY subject_id),
        p AS (SELECT subject_id,bool_and(icd_version='9') AS only9 FROM procedures_icd GROUP BY subject_id)
        SELECT a.subject_id FROM a JOIN d USING(subject_id) LEFT JOIN p USING(subject_id)
        WHERE a.n BETWEEN 2 AND 4 AND a.timed AND d.only9 AND d.n=a.n AND coalesce(p.only9,true)
    """).fetchall()
    selected = sorted(
        (r[0] for r in eligible),
        key=lambda s: hashlib.sha256(
            ("medcpi-real-20261030:" + s).encode()
        ).hexdigest(),
    )[: args.patients]
    if len(selected) < 10:
        raise ValueError(
            "At least 10 eligible patients are required for train/validation/test splitting"
        )
    db.execute("CREATE TEMP TABLE selected(subject_id VARCHAR)")
    db.executemany("INSERT INTO selected VALUES (?)", [(s,) for s in selected])
    counts = {}
    hashes = {}
    for n in tables:
        output = raw / f"{n}.csv"
        query = f"SELECT t.* FROM {n} t JOIN selected s USING(subject_id) ORDER BY ALL"
        output_sql = str(output).replace(chr(92), "/").replace("'", "''")
        db.execute(f"COPY ({query}) TO '{output_sql}' (HEADER, DELIMITER ',')")
        counts[n] = db.execute(
            f"SELECT count(*) FROM {n} t JOIN selected s USING(subject_id)"
        ).fetchone()[0]
        hashes[n] = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "status": "prepared",
        "source": os.environ.get("KG_MODEL_DATASET", "Local MIMIC-IV hospital tables"),
        "selection": f"{len(selected)} fixed-hash selected patients with 2-4 admissions, complete admission times and diagnoses, ICD-9 diagnosis/procedure histories only; all selected admissions preserved",
        "selection_seed": "medcpi-real-20261030",
        "eligible_patients": len(eligible),
        "selected_patients": len(selected),
        "rows": counts,
        "input_sha256": hashes,
    }
    (evidence / "cohort.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "input_sha256"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
