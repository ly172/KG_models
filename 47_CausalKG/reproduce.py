# Generate an artificial table and run the author's R script.

from pathlib import Path
import csv, hashlib, json, math, os, random, subprocess, urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    run = ROOT / "run"
    run.mkdir(exist_ok=True)
    source = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    destination = run / "mediation_analysis.R"
    if not destination.exists():
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            data = response.read()
        assert (
            hashlib.sha256(data).hexdigest() == source["sha256"]
        ), "Author source hash mismatch"
        destination.write_bytes(data)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == source["sha256"]
    rng = random.Random(20261029)
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    rows = []
    for _ in range(2048):
        c1, c2, c3, c4 = [int(rng.random() < p) for p in (0.45, 0.35, 0.5, 0.3)]
        indication = int(rng.random() < sigmoid(-1.0 + 1.5 * c1 + 0.6 * c2 - 0.4 * c3))
        drug = int(
            rng.random() < sigmoid(-1.3 + 1.8 * indication + 0.9 * c1 + 0.7 * c4)
        )
        outcome = int(
            rng.random()
            < sigmoid(-2.0 + 0.7 * indication + 0.9 * drug + 0.7 * c2 + 0.5 * c3)
        )
        rows.append([indication, drug, outcome, c1, c2, c3, c4])
    with (run / "artificial.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["IND", "Drug", "SE", "C1", "C2", "C3", "C4"])
        writer.writerows(rows)
    env = dict(
        os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"
    )
    command = ["Rscript", "--vanilla", "run_author.R", "run/artificial.csv"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (run / "author_stdout.log").write_text(result.stdout, encoding="utf-8")
    (run / "author_stderr.log").write_text(result.stderr, encoding="utf-8")
    print(result.stdout, end="")
    print(result.stderr, end="")
    result.check_returncode()
    with (run / "result.csv").open(encoding="utf-8") as f:
        metrics = next(csv.DictReader(f))
    report = dict(
        status="passed",
        author_exit_code=result.returncode,
        seed=20261029,
        command=command,
        input_kind="2048 artificial rows; no real patients or drug names",
        scope="original published statistical mediation code, including original interaction branch",
        author_script_sha256=source["sha256"],
        input_sha256=hashlib.sha256((run / "artificial.csv").read_bytes()).hexdigest(),
        metrics=metrics,
    )
    (run / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
