#!/usr/bin/env python3
"""Reproduce the teammate CSV schema with bounded memory and a fixed customer split."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from download_energy_data import SGSC_FILES, sha256


def customer_split(columns: list[str], sample_size: int) -> dict[str, list[str]]:
    # Match the teammate's seed-123 customer sampling before fixing their unseeded split.
    selected = np.random.RandomState(123).choice(
        columns, size=min(sample_size, len(columns)), replace=False,
    )
    selected = np.array(sorted(selected, key=int), dtype=object)
    np.random.default_rng(42).shuffle(selected)
    learn_end, fine_end = int(len(selected) * .7), int(len(selected) * .85)
    return {
        "learn": selected[:learn_end].tolist(),
        "finetune": selected[learn_end:fine_end].tolist(),
        "test": selected[fine_end:].tolist(),
    }


def customer_frame(times, readings, customer: str, market: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({
        "DATETIME_30MIN": times, "CUSTOMER_ID": np.int32(customer),
        "KWH_READING": np.asarray(readings, dtype=np.float32),
    }).dropna(subset=["KWH_READING"])
    frame = frame.merge(market, on="DATETIME_30MIN", how="inner", validate="many_to_one")
    frame = frame.sort_values("DATETIME_30MIN")
    timestamp = frame["DATETIME_30MIN"]
    frame["HOUR"] = timestamp.dt.hour.astype("int8")
    frame["DAY_OF_WEEK"] = timestamp.dt.dayofweek.astype("int8")
    frame["IS_WEEKEND"] = (timestamp.dt.dayofweek >= 5).astype("int8")
    frame["MONTH"] = timestamp.dt.month.astype("int8")
    lookup = frame.set_index("DATETIME_30MIN")["KWH_READING"]
    for column, delta in [
        ("KWH_LAG_1HR", timedelta(hours=1)),
        ("KWH_LAG_3DAYS", timedelta(days=3)),
        ("KWH_LAG_1WEEK", timedelta(days=7)),
    ]:
        frame[column] = lookup.reindex(timestamp - delta).to_numpy()
    return frame.dropna()


def prepare(raw_dir: Path, output_dir: Path, sample_size: int = 1500) -> dict:
    if sample_size < 20:
        raise ValueError("Use at least 20 customers to create three useful split groups")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = ["data_learn.csv", "data_finetune.csv", "data_test.csv",
                    "customer_metadata.csv", "preparation_manifest.json"]
    if any((output_dir / name).exists() for name in output_names):
        raise FileExistsError("Prepared outputs already exist; use a new output directory to preserve frozen splits")

    interval = raw_dir / "sgsc" / SGSC_FILES[0][0]
    print("Verifying the 990 MB interval file against the publisher's SHA-256...", flush=True)
    source_hash = sha256(interval)
    if source_hash != SGSC_FILES[0][2]:
        raise ValueError("The interval file is incomplete or does not match the publisher's checksum")
    monthly = [raw_dir / "aemo_monthly" / f"PRICE_AND_DEMAND_2013{m:02d}_NSW1.csv"
               for m in range(1, 13)]
    market = pd.concat([pd.read_csv(path) for path in monthly], ignore_index=True)
    if set(market["REGION"].unique()) != {"NSW1"}:
        raise ValueError("Unexpected AEMO region")
    market = market[["SETTLEMENTDATE", "RRP", "TOTALDEMAND"]].rename(columns={
        "SETTLEMENTDATE": "DATETIME_30MIN", "RRP": "PRICE_AUD_MWH", "TOTALDEMAND": "DEMAND_MW",
    })
    market["DATETIME_30MIN"] = pd.to_datetime(market["DATETIME_30MIN"])
    if market["DATETIME_30MIN"].duplicated().any() or market.isna().any().any():
        raise ValueError("AEMO has missing values or duplicate settlement timestamps")
    market = market.sort_values("DATETIME_30MIN")
    if len(market) != 365 * 48 or not market["DATETIME_30MIN"].diff().dropna().eq(timedelta(minutes=30)).all():
        raise ValueError("AEMO 2013 should contain 17,520 consecutive half-hour settlements")
    market[["PRICE_AUD_MWH", "DEMAND_MW"]] = market[["PRICE_AUD_MWH", "DEMAND_MW"]].astype("float32")

    header = pd.read_csv(interval, nrows=0).columns.tolist()
    split = customer_split([column for column in header if column != "timestamp"], sample_size)
    selected = [customer for group in split.values() for customer in group]
    print(f"Reading only {len(selected)} of {len(header)-1} customer columns...", flush=True)
    wide = pd.read_csv(interval, usecols=["timestamp", *selected],
                       dtype={customer: "float32" for customer in selected})
    times = pd.to_datetime(wide.pop("timestamp"))
    if times.isna().any() or times.duplicated().any():
        raise ValueError("SGSC has invalid or duplicate interval timestamps")

    manifest = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_doi": "10.17632/zm4f727vvr.1", "source_license": "CC BY 4.0",
        "source_interval_sha256": source_hash, "sampling_seed": 123, "split_seed": 42,
        "sampled_customers": len(selected), "aemo_rows": len(market),
        "note": "Same source/schema and 70/15/15 proportions as teammate; fixed split seed, exact-time lags, per-customer processing. No model performance was inspected.",
        "limitations": ["Publisher states missing source readings were filled; the timing of that imputation is not documented here.",
                         "SGSC time labels and AEMO settlement labels are joined exactly as in teammate code; interval start/end and DST alignment require data-owner confirmation. Realised market fields are excluded from forecasting."],
        "splits": {},
        "aemo_source_sha256": {path.name: sha256(path) for path in monthly},
    }
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output_dir) as staging:
        stage = Path(staging)
        for split_name, customers in split.items():
            rows = 0
            first_time, last_time = None, None
            destination = stage / f"data_{split_name}.csv"
            for index, customer in enumerate(customers):
                frame = customer_frame(times, wide[customer], customer, market)
                if frame.empty:
                    raise ValueError(f"No valid rows for selected customer {customer}")
                rows += len(frame)
                start, end = frame["DATETIME_30MIN"].min(), frame["DATETIME_30MIN"].max()
                first_time = start if first_time is None else min(first_time, start)
                last_time = end if last_time is None else max(last_time, end)
                frame.to_csv(destination, mode="w" if index == 0 else "a", header=index == 0, index=False)
                if (index + 1) % 100 == 0 or index + 1 == len(customers):
                    print(f"{split_name}: {index+1}/{len(customers)} customers, {rows:,} rows", flush=True)
            manifest["splits"][split_name] = {
                "customers": len(customers), "customer_ids": customers, "rows": rows,
                "first_timestamp": str(first_time), "last_timestamp": str(last_time),
                "sha256": sha256(destination), "bytes": destination.stat().st_size,
            }

        demographics = pd.read_csv(raw_dir / "sgsc/index files/demographic_lookup_3_groups.csv",
                                   dtype={"customer_ID": "string"})
        metadata = demographics[demographics["customer_ID"].isin(selected)][[
            "customer_ID", "ASSRTD_DWELLING_TYPE_CD", "DWELLING_TYPE_CD", "cluster_house", "cluster_unit",
        ]].rename(columns={"customer_ID": "building_id"})
        # Use the published house/unit clustering assignment, not customer ID or load statistics.
        house = metadata["cluster_house"].gt(0) & metadata["cluster_unit"].eq(0)
        unit = metadata["cluster_unit"].gt(0) & metadata["cluster_house"].eq(0)
        metadata["dwelling_type"] = np.select([house, unit], ["house", "unit"], default="unknown")
        metadata["dwelling_type_source"] = "SGSC published cluster_house/cluster_unit assignment"
        split_by_id = {customer: name for name, customers in split.items() for customer in customers}
        metadata["split"] = metadata["building_id"].map(split_by_id)
        if metadata["building_id"].duplicated().any() or len(metadata) != len(selected):
            raise ValueError("SGSC demographic index does not match selected customers one-to-one")
        metadata.to_csv(stage / "customer_metadata.csv", index=False)
        (stage / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2))
        for name in output_names:
            (stage / name).replace(output_dir / name)
    print(f"Prepared and frozen all three split files in {output_dir}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--sample-size", type=int, default=1500)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.sample_size)
