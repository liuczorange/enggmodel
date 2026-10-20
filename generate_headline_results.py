#!/usr/bin/env python3
"""Write headline-results prose using values read only from results.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from generate_table2 import main as generate_table


HORIZON_LABELS = {"1h": "1 hour", "3d": "3 days", "1w": "1 week"}


def main(results_path: Path, output_path: Path) -> None:
    results = pd.read_csv(results_path)
    required = {"horizon", "model", "macro_mae", "macro_rmse", "pooled_r2",
                "baseline_macro_mae", "macro_mae_improvement_pct", "beats_baseline"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results file is missing columns: {sorted(missing)}")

    paragraphs = [
        "# Headline results",
        "",
        "Table 2 reports errors averaged across test customers, giving each household equal "
        "weight regardless of its number of scored observations. Each machine-learning model is compared "
        "with the free historical guess matched to its forecast distance.",
        "",
    ]
    order = [h for h in ("1h", "3d", "1w") if h in set(results["horizon"])]
    for horizon in order:
        horizon_rows = results[results["horizon"] == horizon]
        candidates = horizon_rows[~horizon_rows["model"].str.startswith("naive_")]
        best = candidates.loc[candidates["macro_mae"].idxmin()]
        baseline = float(best["baseline_macro_mae"])
        improvement = float(best["macro_mae_improvement_pct"])
        comparison = "beat" if float(best["macro_mae"]) < baseline else "did not beat"
        paragraphs.append(
            f"At **{HORIZON_LABELS[horizon]}**, **{best['model']}** was the stronger "
            f"machine-learning model (macro MAE **{best['macro_mae']:.3f} kWh**). "
            f"It {comparison} the matching free guess "
            f"(macro MAE **{baseline:.3f} kWh**), a change of **{improvement:.1f}%**."
        )
        paragraphs.append("")

    if len(order) > 1:
        best_by_horizon = []
        for horizon in order:
            candidates = results[
                (results["horizon"] == horizon)
                & ~results["model"].str.startswith("naive_")
            ]
            best_by_horizon.append(candidates.loc[candidates["macro_mae"].idxmin()])
        sequence = ", ".join(
            f"{HORIZON_LABELS[row['horizon']]}: {row['macro_mae']:.3f} kWh"
            for row in best_by_horizon
        )
        paragraphs.append(
            "Across the three distances, the best machine-learning macro MAE values were "
            f"{sequence}. This sequence should be interpreted together with each distance's "
            "own free guess rather than against zero error."
        )
        paragraphs.append("")

    table_path = output_path.with_name("table2.md")
    generate_table(results_path, table_path)
    paragraphs.extend(["## Table 2", "", table_path.read_text().strip(), ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(paragraphs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("multihorizon_outputs/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("multihorizon_outputs/headline_results.md"))
    args = parser.parse_args()
    main(args.results, args.output)
