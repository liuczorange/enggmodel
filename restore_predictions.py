#!/usr/bin/env python3
"""Restore the complete published prediction exports and check their hashes."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "multihorizon_outputs"
    expected = json.loads((output / "verification_summary.json").read_text())["artifact_sha256"]
    for name in ("predictions.csv", "predictions_wide.csv"):
        destination = output / name
        if destination.exists():
            if sha256(destination) != expected[name]:
                raise ValueError(f"Existing file differs from the published experiment; not overwritten: {destination}")
            print(f"Already restored and verified: {name}")
            continue
        archive = root / "artifacts" / (name + ".gz")
        with tempfile.NamedTemporaryFile(dir=output, prefix=name + ".restoring-", delete=False) as temporary:
            staging = Path(temporary.name)
        try:
            with gzip.open(archive, "rb") as source, staging.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if sha256(staging) != expected[name]:
                raise ValueError(f"Restored file failed the published checksum: {name}")
            staging.replace(destination)
            print(f"Restored and verified: {name}")
        finally:
            staging.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
