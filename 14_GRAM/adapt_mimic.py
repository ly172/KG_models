# Select ICD-9 admissions and write them in the author's input format.

import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path


def table_path(root, module, name):
    base = Path(root) / module
    candidates = [
        base / (name + ".csv") / (name + ".csv"),
        base / (name + ".csv"),
        base / (name + ".csv.gz"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("MIMIC table missing: " + module + "/" + name)


def open_table(path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", newline="")
    )


def columns(stream, names):
    reader = csv.reader(stream)
    header = next(reader)
    indices = [header.index(name) for name in names]
    for row in reader:
        yield tuple(row[index] for index in indices)


def adapt(root, output, patient_limit=160, selection_seed=20261029):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    admissions_path = table_path(root, "hosp", "admissions")
    diagnoses_path = table_path(root, "hosp", "diagnoses_icd")
    patients = collections.defaultdict(list)
    admissions = {}
    print("Reading original admissions table...", flush=True)
    with open_table(admissions_path) as stream:
        for subject, admission, admission_time in columns(
            stream, ["subject_id", "hadm_id", "admittime"]
        ):
            sid, hadm = int(subject), int(admission)
            admissions[hadm] = (sid, admission_time)
            patients[sid].append(hadm)
    visits9 = set()
    patients_other_codes = set()
    code_versions = collections.Counter()
    print(
        "Checking diagnosis code versions for complete patient histories...", flush=True
    )
    with open_table(diagnoses_path) as stream:
        for subject, admission, version, code in columns(
            stream, ["subject_id", "hadm_id", "icd_version", "icd_code"]
        ):
            code_versions[version] += 1
            sid, hadm = int(subject), int(admission)
            if version == "9" and code.strip():
                visits9.add(hadm)
            else:
                patients_other_codes.add(sid)
    eligible = [
        sid
        for sid, visits in patients.items()
        if 2 <= len(visits) <= 4
        and sid not in patients_other_codes
        and all(hadm in visits9 and admissions[hadm][1] for hadm in visits)
    ]
    eligible.sort(
        key=lambda sid: hashlib.sha256(
            ("GRAM:%d:%d" % (selection_seed, sid)).encode()
        ).digest()
    )
    selected = eligible[:patient_limit]
    if int(len(selected) * 0.1) < 1 or int(len(selected) * 0.2) < 1:
        raise RuntimeError(
            "At least 10 eligible patients are needed for the author 70/10/20 split"
        )
    selected_set = set(selected)
    selected_visits = {hadm for sid in selected for hadm in patients[sid]}
    print(
        "Fixed cohort selected: %d patients, %d admissions. Reading their original diagnosis rows..."
        % (len(selected), len(selected_visits)),
        flush=True,
    )
    admission_output = output / "ADMISSIONS.csv"
    diagnosis_output = output / "DIAGNOSES_ICD.csv"
    with admission_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["ROW_ID", "SUBJECT_ID", "HADM_ID", "ADMITTIME"])
        row_id = 0
        for sid in selected:
            for hadm in sorted(
                patients[sid], key=lambda value: (admissions[value][1], value)
            ):
                row_id += 1
                # The original parser expects an unquoted timestamp in column 4.
                stream.write("%d,%d,%d,%s\n" % (row_id, sid, hadm, admissions[hadm][1]))
    selected_codes = set()
    diagnoses_written = 0
    with open_table(diagnoses_path) as source, diagnosis_output.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.writer(target, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        for subject, admission, version, code, sequence in columns(
            source, ["subject_id", "hadm_id", "icd_version", "icd_code", "seq_num"]
        ):
            sid, hadm = int(subject), int(admission)
            if sid in selected_set and hadm in selected_visits:
                assert version == "9"
                diagnoses_written += 1
                code = code.strip()
                selected_codes.add(code)
                writer.writerow([diagnoses_written, sid, hadm, int(sequence), code])
    summary = {
        "source": os.environ.get("KG_MODEL_DATASET", "Local MIMIC-IV hospital tables"),
        "source_tables": {
            "admissions": admissions_path.name,
            "diagnoses_icd": diagnoses_path.name,
        },
        "source_file_bytes": {
            "admissions": admissions_path.stat().st_size,
            "diagnoses_icd": diagnoses_path.stat().st_size,
        },
        "source_diagnosis_rows_by_version": dict(code_versions),
        "selection_rule": "Patient has 2-4 total hospital admissions, every admission has nonempty ICD-9 diagnoses and admission time, and no diagnosis record of another code version. Select by SHA256(GRAM:seed:subject_id), before training.",
        "selection_seed": selection_seed,
        "eligible_patients": len(eligible),
        "selected_patients": len(selected),
        "selected_admissions": len(selected_visits),
        "selected_diagnosis_rows": diagnoses_written,
        "selected_distinct_full_icd9_codes": len(selected_codes),
    }
    (output / "aggregate_input.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return admission_output, diagnosis_output, summary
