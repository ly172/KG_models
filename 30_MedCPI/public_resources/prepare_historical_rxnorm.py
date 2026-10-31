# Download two older RxNorm Prescribable releases.
# Run prepare_rxnorm.py first with the same output folder.

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

RELEASES = [
    dict(
        release="2014-12-01",
        filename="RxNorm_full_prescribe_12012014.zip",
        sha256="348b68206229a4a9b26b9917b4cc65f9860f6885776485da74d926aaee56b968",
        expected_bytes=51919060,
    ),
    dict(
        release="2019-12-02",
        filename="RxNorm_full_prescribe_12022019.zip",
        sha256="5bd1cb752eb4091ff9241a1dee1c49fde5aca1dca4c9c38c9803dac6faee4ded",
        expected_bytes=72944795,
    ),
]


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for item in RELEASES:
        url = "https://download.nlm.nih.gov/rxnorm/" + item["filename"]
        archive = root / item["filename"]
        if not archive.exists():
            partial = archive.with_suffix(archive.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=180) as response, partial.open(
                "wb"
            ) as target:
                shutil.copyfileobj(response, target)
            if (
                partial.stat().st_size != item["expected_bytes"]
                or digest(partial, "sha256") != item["sha256"]
            ):
                raise ValueError(
                    "Downloaded release failed its pinned byte/hash check: "
                    + item["release"]
                )
            partial.replace(archive)
        if (
            archive.stat().st_size != item["expected_bytes"]
            or digest(archive, "sha256") != item["sha256"]
        ):
            raise ValueError(
                "Existing archive failed its pinned byte/hash check: " + str(archive)
            )
        folder = root / ("rxnorm_prescribable_" + item["release"].replace("-", ""))
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            if source.testzip() is not None:
                raise ValueError("ZIP integrity check failed: " + str(archive))
            for entry in source.infolist():
                basename = Path(entry.filename).name
                if not entry.is_dir() and (
                    basename.endswith(".RRF") or basename.lower().startswith("readme")
                ):
                    with source.open(entry) as content, (folder / basename).open(
                        "wb"
                    ) as target:
                        shutil.copyfileobj(content, target)
        receipt = dict(
            release=item["release"],
            url=url,
            bytes=archive.stat().st_size,
            sha256=item["sha256"],
            md5=digest(archive, "md5"),
            directory=str(folder),
        )
        receipts.append(receipt)
        print(json.dumps(receipt))
    (root / "rxnorm_historical_downloads.json").write_text(
        json.dumps({"releases": receipts}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
