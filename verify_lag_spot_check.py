#!/usr/bin/env python3
"""Cross-check one fine-tuning household directly against the downloaded source."""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def main(raw_dir: Path, output_dir: Path) -> None:
    check = pd.read_csv(output_dir / "manual_lag_check_finetune.csv", dtype={"CUSTOMER_ID": "string"})
    if check.empty or check["CUSTOMER_ID"].nunique() != 1:
        raise ValueError("Expected spot-check rows for exactly one fine-tuning customer")
    customer = check["CUSTOMER_ID"].iloc[0]
    raw = pd.read_csv(raw_dir / "sgsc/interval load data/data_3_groups.csv",
                      usecols=["timestamp", customer], parse_dates=["timestamp"])
    source = raw.set_index("timestamp")[customer]
    rows = []
    lags = [("1h", timedelta(hours=1), "KWH_LAG_1HR", "lag_1h"),
            ("3d", timedelta(days=3), "KWH_LAG_3DAYS", "lag_3d"),
            ("1w", timedelta(days=7), "KWH_LAG_1WEEK", "lag_1w")]
    for _, target in check.iterrows():
        target_time = pd.Timestamp(target["DATETIME_30MIN"])
        for horizon, delta, provided, rebuilt in lags:
            source_time = target_time - delta
            raw_value = float(source.loc[source_time])
            supplied = float(target[provided])
            computed = float(target[rebuilt])
            rows.append({
                "building_id": customer, "target_timestamp": target_time,
                "horizon": horizon, "forecast_origin": source_time,
                "source_reading_timestamp": source_time, "raw_source_kwh": raw_value,
                "prepared_lag_kwh": supplied, "model_lag_kwh": computed,
                "source_minus_model_kwh": raw_value - computed,
                "agrees_with_source": bool(np.isclose(raw_value, supplied, atol=1e-6, rtol=1e-5)
                                           and np.isclose(raw_value, computed, atol=1e-6, rtol=1e-5)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "raw_source_lag_spot_check.csv", index=False)
    if not result["agrees_with_source"].all():
        raise ValueError("A prepared/model lag does not match the original source")
    print(f"Source cross-check passed: customer {customer}, {len(result)} target/lag comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("multihorizon_outputs"))
    args = parser.parse_args()
    main(args.raw_dir, args.output_dir)
