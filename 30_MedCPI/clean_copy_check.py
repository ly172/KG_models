# Copy the code to an empty folder and rerun all stages.

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run_author_stages import environment

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", type=Path, required=True)
    ap.add_argument("--public-cache", type=Path, required=True)
    ap.add_argument("--sapbert-cache", type=Path, required=True)
    ap.add_argument("--compare-existing", action="store_true")
    ap.add_argument("--patients", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    copy_root = ROOT / ".private/clean_repository"
    target = copy_root / "30_MedCPI"
    names = [p.name for p in ROOT.glob("*.py") if p.name != "clean_copy_check.py"] + [
        "pixi.toml",
        "pixi.lock",
        ".gitignore",
        "author_source.json",
    ]
    if not a.compare_existing:
        target.mkdir(parents=True, exist_ok=False)
        shutil.copytree(
            next(
                parent / "shared"
                for parent in ROOT.parents
                if (parent / "shared/downloads.py").is_file()
            ),
            copy_root / "shared",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        for name in names:
            shutil.copyfile(ROOT / name, target / name)
        shutil.copytree(
            ROOT / "public_resources",
            target / "public_resources",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (target / "verification").mkdir()
        shutil.copyfile(
            ROOT / "verification/sapbert_source.json",
            target / "verification/sapbert_source.json",
        )
        log = ROOT / ".private/logs/clean_copy.log"
        # The child process uses paths inside the copy.
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.run(
                [
                    sys.executable,
                    str(target / "reproduce.py"),
                    "--mimic-root",
                    str(a.mimic_root),
                    "--public-cache",
                    str(a.public_cache),
                    "--sapbert-cache",
                    str(a.sapbert_cache),
                    "--patients",
                    str(a.patients),
                    "--epochs",
                    str(a.epochs),
                ],
                cwd=target,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        exit_code = p.returncode
    else:
        exit_code = json.loads(
            (ROOT / "verification/clean_copy.json").read_text(encoding="utf-8")
        )["exit_code"]
    result = {
        "exit_code": exit_code,
        "verified_source_files": len(names),
        "cached_inputs": "SHA256-verified public terminology downloads and public SapBERT weights only",
        "environment": "Shared already locked Python environment; independent code/output directory",
        "comparisons": {},
    }
    if exit_code:
        (ROOT / "verification/clean_copy.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        raise RuntimeError("Independent run failed; private log retained")
    for name in [
        "cohort.json",
        "public_input_projection.json",
        "mapping.json",
        "pipeline.json",
        "training.json",
    ]:
        old = json.loads((ROOT / "verification" / name).read_text(encoding="utf-8"))
        new = json.loads((target / "verification" / name).read_text(encoding="utf-8"))
        if name == "cohort.json":
            old.pop("private_input_sha256", None)
            new.pop("private_input_sha256", None)
            old.pop("input_sha256", None)
            new.pop("input_sha256", None)

            # Absolute paths differ between copies.
            def scrub(x):
                if isinstance(x, dict):
                    return {
                        k: scrub(v)
                        for k, v in x.items()
                        if "path" not in k and "root" not in k
                    }
                if isinstance(x, list):
                    return [scrub(v) for v in x]
                return x

            old = scrub(old)
            new = scrub(new)
        if name == "mapping.json":
            for value in old.values():
                if isinstance(value, dict):
                    value.pop("seconds", None)
            for value in new.values():
                if isinstance(value, dict):
                    value.pop("seconds", None)
        result["comparisons"][name] = old == new
    table_comparison = {}
    for table in ["admissions", "diagnoses_icd", "procedures_icd", "prescriptions"]:
        paths = [base / f".private/raw/mimic4/{table}.csv" for base in [ROOT, target]]
        rows = [
            list(csv.reader(path.open(encoding="utf-8", newline=""))) for path in paths
        ]
        table_comparison[table] = {
            "headers_identical": rows[0][0] == rows[1][0],
            "all_fields_and_row_multiplicities_identical": Counter(
                map(tuple, rows[0][1:])
            )
            == Counter(map(tuple, rows[1][1:])),
            "row_order_identical": rows[0][1:] == rows[1][1:],
            "file_bytes_identical": hashlib.sha256(paths[0].read_bytes()).digest()
            == hashlib.sha256(paths[1].read_bytes()).digest(),
        }
    result["raw_table_comparison"] = table_comparison
    result["raw_table_interpretation"] = (
        "CSV physical row ordering may differ. Every field and row multiplicity is compared; downstream mappings, graphs, training and model tensors are also compared."
    )
    result["final_source_files_match_independent_directory"] = all(
        (ROOT / name).read_bytes() == (target / name).read_bytes() for name in names
    )
    import torch

    def checkpoint(base):
        return torch.load(
            base
            / ".private/outputs/integrate/mimic4/mortality/best_valid_formal_report_fixed_seed0.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]

    original = checkpoint(ROOT)
    repeated = checkpoint(target)
    result["all_model_tensors_identical"] = all(
        torch.equal(v, repeated[k]) for k, v in original.items()
    )
    (ROOT / "verification/clean_copy.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if (
        not all(result["comparisons"].values())
        or not result["all_model_tensors_identical"]
        or not result["final_source_files_match_independent_directory"]
        or not all(
            v["headers_identical"] and v["all_fields_and_row_multiplicities_identical"]
            for v in table_comparison.values()
        )
    ):
        raise AssertionError("Independent-copy results differ; inspect comparisons")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
