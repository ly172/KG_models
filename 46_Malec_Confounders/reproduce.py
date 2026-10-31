# Build PSI vectors from generated triples and run the author's query.

from pathlib import Path
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def matches(path, source):
    if "content_sha256" not in source:
        return hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    try:
        with zipfile.ZipFile(path) as bundle:
            rows = sorted(
                [
                    info.filename.split("/", 1)[1],
                    hashlib.sha256(bundle.read(info)).hexdigest(),
                ]
                for info in bundle.infolist()
                if not info.is_dir()
            )
    except (zipfile.BadZipFile, IndexError):
        return False
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    return len(rows) == source["files"] and digest == source["content_sha256"]


def prepare():
    import truststore

    for name, source in json.loads((ROOT / "sources.json").read_text()).items():
        target = ROOT / "cache" / name
        target.parent.mkdir(exist_ok=True, parents=True)
        if target.exists() and matches(target, source):
            continue
        temp = target.with_suffix(target.suffix + ".download")
        try:
            with urllib.request.urlopen(
                source["url"],
                context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                timeout=120,
            ) as response, temp.open("wb") as f:
                shutil.copyfileobj(response, f)
        except Exception:
            if os.name != "nt":
                raise
            subprocess.run(
                [
                    "curl.exe",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--output",
                    str(temp),
                    source["url"],
                ],
                check=True,
            )
        if not matches(temp, source):
            raise RuntimeError("Source checksum differs: " + name)
        temp.replace(target)


def main():
    prepare()
    java = shutil.which("java")
    if not java:
        raise RuntimeError(
            "Run with pixi run --locked smoke to use the pinned Java runtime"
        )
    cache = ROOT / "cache"
    src = cache / "semanticvectors-f585f0a66973b63cc6c0952e3fa9610825f285a3"
    maven = cache / "apache-maven-3.9.9"
    if not src.exists():
        with zipfile.ZipFile(cache / "semanticvectors.zip") as z:
            z.extractall(cache)
    if not maven.exists():
        with zipfile.ZipFile(cache / "maven.zip") as z:
            z.extractall(cache)
    jar = src / "target/semanticvectors-5.9.jar"
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    if not jar.exists():
        command = [
            java,
            "-Dmaven.home=" + str(maven),
            "-Dmaven.multiModuleProjectDirectory=" + str(src),
            "-Dclassworlds.conf=" + str(maven / "bin/m2.conf"),
            "-classpath",
            str(next((maven / "boot").glob("plexus-classworlds-*.jar"))),
            "org.codehaus.plexus.classworlds.launcher.Launcher",
            "-B",
            "-ntp",
            "-Dmaven.repo.local=" + str(cache / "maven_repository"),
            "-f",
            str(src / "pom.xml"),
            "package",
            "-P",
            "endUserRelease",
        ]
        print("Building SemanticVectors 5.9 from fixed upstream source", flush=True)
        with (out / "build.log").open("w", encoding="utf-8") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    work = ROOT / "work" / str(time.time_ns())
    work.mkdir(parents=True)
    triples = [
        ("syn_druga", "TREATS", "syn_conditiona"),
        ("syn_druga", "TREATS", "syn_conditionb"),
        ("syn_druga", "TREATS", "syn_conditionc"),
        ("syn_drugb", "TREATS", "syn_conditiond"),
        ("syn_conditiona", "CAUSES", "syn_outcome"),
        ("syn_conditionb", "CAUSES", "syn_outcome"),
        ("syn_conditiond", "CAUSES", "syn_outcome"),
        ("syn_conditionc", "CAUSES", "syn_otheroutcome"),
        ("syn_druge", "TREATS", "syn_conditione"),
        ("syn_conditione", "CAUSES", "syn_otheroutcome"),
        ("syn_conditionf", "CAUSES", "syn_otheroutcome"),
        ("syn_drugf", "TREATS", "syn_conditionf"),
    ]
    (work / "triples.tsv").write_text(
        "\n".join("\t".join(t) for t in triples) + "\n", encoding="utf-8"
    )
    commands = []

    def run(name, args):
        p = subprocess.run(
            [java, "-cp", str(jar)] + args,
            cwd=work,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        (out / (name + ".log")).write_text(p.stdout + p.stderr, encoding="utf-8")
        p.check_returncode()
        commands.append(
            {"entry": args[0], "arguments": args[1:], "exit_code": p.returncode}
        )
        return p.stdout

    run("index", ["pitt.search.lucene.LuceneIndexFromTriples", "triples.tsv"])
    run(
        "psi",
        [
            "pitt.search.semanticvectors.PSI",
            "-dimension",
            "32000",
            "-vectortype",
            "binary",
            "-termweight",
            "idf",
            "-luceneindexpath",
            "predication_index",
        ],
    )
    vectors = {
        name: (work / name).stat().st_size
        for name in [
            "elementalvectors.bin",
            "semanticvectors.bin",
            "predicatevectors.bin",
        ]
    }
    query = "S(syn_druga)*P(TREATS)+S(syn_outcome)*P(CAUSES-INV)"
    original = (cache / "getConfounders.sh").read_text()
    assert "S($1)*P(TREATS)+S($2)*P(CAUSES-INV)" in original
    text = run(
        "search",
        [
            "pitt.search.semanticvectors.Search",
            "-elementalvectorfile",
            "elementalvectors.bin",
            "-semanticvectorfile",
            "semanticvectors.bin",
            "-elementalpredicatevectorfile",
            "predicatevectors.bin",
            "-matchcase",
            "-searchvectorfile",
            "elementalvectors.bin",
            "-searchtype",
            "boundproduct",
            "-numsearchresults",
            "5",
            query,
        ],
    )
    predictions = [
        {"score": float(line.split(":", 1)[0]), "concept": line.split(":", 1)[1]}
        for line in text.splitlines()
        if ":" in line
    ]
    assert len(predictions) == 5 and {p["concept"] for p in predictions[:2]} == {
        "syn_conditiona",
        "syn_conditionb",
    }
    result = {
        "status": "component_passed",
        "scope": "original PSI construction and author query on generated triples",
        "triples": len(triples),
        "vector_parameters": {
            "dimension": 32000,
            "vectortype": "binary",
            "termweight": "idf",
        },
        "vectors": vectors,
        "predictions": predictions,
        "commands": commands,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(work / "triples.tsv", out / "triples.tsv")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
