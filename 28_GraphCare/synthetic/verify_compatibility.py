# Record the missing dependency and compare model definitions.

import json
import os
import subprocess
import sys
from prepare_author import ROOT, prepare


def main():
    output = ROOT / "verification/compatibility"
    output.mkdir(parents=True, exist_ok=True)
    original, _ = prepare(repair_imports=False, output=ROOT / ".run/probe_original")
    runtime, checks = prepare()
    results = []
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONIOENCODING="utf-8",
    )
    for name, cwd, success in [
        ("original_import", original, False),
        ("repaired_import", runtime, True),
    ]:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "from graphcare_ import GraphCare; from graphcare import train, evaluate",
            ],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        (output / (name + ".log")).write_text(
            process.stdout.replace(str(ROOT), "<PACKAGE_ROOT>"), encoding="utf-8"
        )
        results.append(
            {
                "case": name,
                "exit_code": process.returncode,
                "expected_success": success,
                "expectation_met": (process.returncode == 0) == success,
            }
        )
    result = {
        "cases": results,
        "core_source_checks": checks,
        "log_edit": "Only package absolute path replaced with <PACKAGE_ROOT>; errors retained.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert all(item["expectation_met"] for item in results)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
