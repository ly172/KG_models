# Download public files with retries and resume.

import hashlib
import http.client
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def download_file(
    url, target, expected_sha256, expected_bytes=None, cache=None, attempts=4
):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    def valid(path):
        return (
            path.is_file()
            and (expected_bytes is None or path.stat().st_size == expected_bytes)
            and sha256(path) == expected_sha256
        )

    if valid(target):
        return target
    if cache is not None and valid(Path(cache)):
        shutil.copyfile(cache, target)
        return target
    partial = target.with_name(target.name + ".part")
    size = (
        " (%.1f MB)" % (expected_bytes / 1000000) if expected_bytes is not None else ""
    )
    print("Downloading public file: %s%s" % (target.name, size), flush=True)
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_bytes is not None and offset >= expected_bytes:
            partial.unlink()
            offset = 0
        headers = {
            "User-Agent": "kg_model-public-download",
            "Accept-Encoding": "identity",
        }
        if offset:
            headers["Range"] = "bytes=%d-" % offset
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(
                    "bytes %d-" % offset
                ):
                    raise IOError("Server returned an unexpected byte range")
                with partial.open("ab" if append else "wb") as stream:
                    written = offset if append else 0
                    last_progress = time.monotonic()
                    while True:
                        chunk = response.read(1048576)
                        if not chunk:
                            break
                        stream.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_progress >= 20:
                            print(
                                "  %s: %.1f MB received"
                                % (target.name, written / 1000000),
                                flush=True,
                            )
                            last_progress = time.monotonic()
            if not valid(partial):
                # Resume a partial download, otherwise start over.
                if expected_bytes is None or partial.stat().st_size >= expected_bytes:
                    partial.unlink()
                raise IOError("Downloaded file did not pass its size and SHA256 check")
            partial.replace(target)
            return target
        except (
            OSError,
            urllib.error.URLError,
            http.client.HTTPException,
            ValueError,
        ) as error:
            if (
                isinstance(error, urllib.error.HTTPError)
                and error.code == 416
                and partial.exists()
            ):
                partial.unlink()
            if attempt + 1 == attempts:
                raise RuntimeError(
                    "Download failed after %d attempts: %s" % (attempts, url)
                ) from error
            print(
                "Retrying public download (%d/%d): %s"
                % (attempt + 2, attempts, target.name),
                flush=True,
            )
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("Download attempts exhausted")
