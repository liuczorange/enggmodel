#!/usr/bin/env python3
"""Generate report-ready Table 2 strictly from results.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main(results_path: Path, output_path: Path) -> None:
    results = pd.read_csv(results_path)
    required = {"horizon", "model", "macro_mae", "macro_rmse", "pooled_r2",
                "macro_mae_improvement_pct", "beats_baseline"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results file is missing columns: {sorted(missing)}")
    table = results[[
        "horizon", "model", "macro_mae", "macro_rmse", "pooled_r2",
        "macro_mae_improvement_pct", "beats_baseline",
    ]].copy()
    table.columns = [
        "Forecast distance", "Model", "Macro MAE (kWh)", "Macro RMSE (kWh)",
        "Pooled R²", "MAE improvement vs free guess (%)", "Beats free guess",
    ]
    for col in ["Macro MAE (kWh)", "Macro RMSE (kWh)", "Pooled R²",
                "MAE improvement vs free guess (%)"]:
        table[col] = table[col].round(3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [str(column) for column in table.columns]
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        values = [
            ("—" if pd.isna(value) else str(value)).replace("|", "\\|")
            for value in row
        ]
        markdown.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(markdown) + "\n")
    table.to_csv(output_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("multihorizon_outputs/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("multihorizon_outputs/table2.md"))
    args = parser.parse_args()
    main(args.results, args.output)
