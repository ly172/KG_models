# Record import and device errors and compare model definitions.

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_author import ROOT, prepare


def definition_hashes(text):
    return {
        node.name: hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()
        for node in ast.parse(text).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def main():
    output = ROOT / "verification/compatibility"
    output.mkdir(parents=True, exist_ok=True)
    original = prepare(ROOT / ".run/probe_original", repair_imports=False, cpu=False)
    imports_fixed = prepare(ROOT / ".run/probe_imports", repair_imports=True, cpu=False)
    runtime = prepare()
    cases = [
        ("original_dnc_import", original / "code", "import models", False),
        ("original_removed_metric_import", original / "code", "import util", False),
        (
            "original_cuda_device",
            original / "code",
            "import torch; torch.zeros(1).to(torch.device('cuda:0'))",
            False,
        ),
        (
            "original_legacy_vocabulary",
            original / "code",
            "import dill; dill.load(open('../data/voc_final.pkl', 'rb'))",
            False,
        ),
        ("repaired_imports", runtime / "code", "import models; import util", True),
    ]
    results = []
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONIOENCODING="utf-8",
    )
    for name, cwd, code, expect_success in cases:
        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        log = process.stdout.replace(str(ROOT), "<PACKAGE_ROOT>")
        (output / (name + ".log")).write_text(log, encoding="utf-8")
        results.append(
            {
                "case": name,
                "exit_code": process.returncode,
                "expected_success": expect_success,
                "expectation_met": (process.returncode == 0) == expect_success,
            }
        )
    source_checks = {}
    for rel in ["code/models.py", "code/layers.py", "code/util.py"]:
        before = definition_hashes(
            (ROOT / "upstream" / rel).read_text(encoding="utf-8")
        )
        after = definition_hashes((runtime / rel).read_text(encoding="utf-8"))
        source_checks[rel] = {
            "all_function_and_class_definitions_identical": before == after,
            "definition_count": len(before),
        }
    report = {
        "cases": results,
        "source_checks": source_checks,
        "log_edit": "Only package absolute path replaced with <PACKAGE_ROOT>; errors retained.",
        "training_schedule_change": "Run entry sets CPU and requested epochs; author default epochs 40 remains when environment override is absent.",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert all(case["expectation_met"] for case in results)
    assert all(
        value["all_function_and_class_definitions_identical"]
        for value in source_checks.values()
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
