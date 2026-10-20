#!/usr/bin/env python3
"""Download the public SGSC and AEMO sources without loading them into RAM."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess


SGSC_BASE = "https://data.mendeley.com/public-files/datasets/zm4f727vvr/files"
SGSC_FILES = [
    ("interval load data/data_3_groups.csv", "32c3d6e7-0602-48c0-98f9-6ff467f1b1e0",
     "cc152fec85d33710829a149173d8e98f09e0e47f0fcc423817dec1882fc30a6d"),
    ("index files/cluster_index.csv", "7e806e5d-4655-4542-9f0f-6e3b71af0347",
     "c41445977e2da32580b15b98e7561aa602d437a2f59f81d94ad931e7d77e5143"),
    ("index files/Column Mappings.csv", "7fb18841-6c4b-47c8-89df-7a58a59c11ac",
     "cadeeeadf3f58436f26a7be4d18e0074de519ec6987450b2f8f736b0b67c93cb"),
    ("index files/demographic_lookup_3_groups.csv", "45c506b0-2dd3-4638-9bf1-317d955778f8",
     "53c0bdc6825b4122843ac5dfc6ced25e6ee4eae14b53513694b75f713785bbe6"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, expected_hash: str | None = None) -> None:
    if path.exists() and (expected_hash is None or sha256(path) == expected_hash):
        print(f"Already downloaded: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading: {path}", flush=True)
    subprocess.run([
        "curl", "-L", "--fail", "--silent", "--show-error", "--retry", "3",
        "--connect-timeout", "30", "--max-time", "1800", "-C", "-",
        "-A", "Mozilla/5.0", url, "-o", str(partial),
    ], check=True)
    if expected_hash and sha256(partial) != expected_hash:
        raise ValueError(f"Downloaded file failed official SHA-256 check: {partial}")
    partial.replace(path)
    print(f"Downloaded and checked: {path}", flush=True)


def main(raw_dir: Path, support_only: bool) -> None:
    records = SGSC_FILES[1:] if support_only else SGSC_FILES
    for relative, file_id, expected in records:
        download(f"{SGSC_BASE}/{file_id}/file_downloaded", raw_dir / "sgsc" / relative, expected)

    def fetch_month(month):
        name = f"PRICE_AND_DEMAND_2013{month:02d}_NSW1.csv"
        download(f"https://aemo.com.au/aemo/data/nem/priceanddemand/{name}",
                 raw_dir / "aemo_monthly" / name)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch_month, range(1, 13)))
    print("All requested public source files are available.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--support-only", action="store_true",
                        help="Download indexes and AEMO while the large interval file downloads separately")
    args = parser.parse_args()
    main(args.raw_dir, args.support_only)
