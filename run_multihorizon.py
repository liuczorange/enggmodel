#!/usr/bin/env python3
"""Train and evaluate leakage-safe 1-hour, 3-day, and 1-week forecasts."""
from __future__ import annotations

import argparse
from pathlib import Path

from multihorizon.config import HORIZONS
from multihorizon.pipeline import final_test, tune


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"),
                        help="Directory containing data_learn.csv, data_finetune.csv, data_test.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("multihorizon_outputs"))
    parser.add_argument("--horizon", choices=[*HORIZONS, "all"], default="all",
                        help="The one forecasting-distance dial")
    parser.add_argument("--stage", choices=["tune", "final"], required=True,
                        help="Tune never reads test; final is locked after one successful run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "tune":
        tune(args.data_dir, args.output_dir, args.horizon)
        print(f"Tuning complete. Settings: {args.output_dir / 'selected_settings.json'}")
    else:
        final_test(args.data_dir, args.output_dir, args.horizon)
        print(f"Final test complete. Results: {args.output_dir / 'results.csv'}")
