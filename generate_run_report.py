#!/usr/bin/env python3
"""Assemble report sections and analysis handover from the saved, locked experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from generate_headline_results import main as generate_headlines


def main(output_dir: Path, data_dir: Path) -> None:
    if not (output_dir / "FINAL_TEST_COMPLETE.json").exists():
        raise FileNotFoundError("A completed, locked final test is required before report generation")
    generate_headlines(output_dir / "results.csv", output_dir / "headline_results.md")
    cohort = pd.read_csv(output_dir / "cohort.csv", dtype={"building_id": "string"})
    metadata = pd.read_csv(data_dir / "customer_metadata.csv", dtype={"building_id": "string"})
    per_building = pd.read_csv(output_dir / "per_building_metrics.csv", dtype={"building_id": "string"})
    joined = per_building.merge(metadata, on="building_id", how="left", validate="many_to_one")
    if joined["dwelling_type"].isna().any():
        raise ValueError("A tested customer has no dwelling metadata")
    joined.to_csv(output_dir / "per_building_metrics_with_type.csv", index=False)
    type_errors = joined.groupby(["horizon", "model", "dwelling_type"], as_index=False).agg(
        n_buildings=("building_id", "nunique"), macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"),
    )
    type_errors.to_csv(output_dir / "error_by_dwelling_type.csv", index=False)

    settings = json.loads((output_dir / "selected_settings.json").read_text())
    runtime = json.loads((output_dir / "runtime_versions.json").read_text())
    trials = pd.read_csv(output_dir / "validation_trials.csv")
    counts = cohort.groupby("split").size().to_dict()
    root = Path(__file__).resolve().parent
    draft = (root / "docs/modelling_methods_draft.md").read_text()
    methods = "# Modelling methods\n\n" + draft[draft.index("## Forecasting task"):]
    methods = methods.replace(
        "## Models and tuning\n",
        "## Models and tuning\n\n"
        f"For a reproducible laptop-scale experiment, a fixed random sample of {counts['learn']} "
        f"learning, {counts['finetune']} fine-tuning and {counts['test']} held-out test customers "
        "was drawn from the prepared customer-disjoint split. These caps were fixed before model "
        "selection. The selected customer IDs and random seeds are retained; conclusions apply to "
        "this cohort rather than the entire source population.\n",
    )
    methods += (
        "\n## Source-data limitations\n\n"
        "The publisher has already filled missing SGSC readings. The original imputation may "
        "have used later observations, so the experiment is a retrospective evaluation on a "
        "processed dataset. Learning and test households cover the same calendar year; "
        "this evaluates transfer to unseen households, not a chronological future-year holdout. "
        "It does not establish prospective performance on a live stream. The original "
        "SGSC/AEMO timestamp join is retained; interval-start/end and daylight-saving alignment "
        "remain upstream assumptions. Realised market fields are excluded from prediction. "
        "Macro averaging gives each household equal weight, but it does not normalise away "
        "differences in consumption scale.\n"
    )
    (output_dir / "modelling_methods.md").write_text(methods)

    appendix = [
        "# Appendix: executed model settings", "",
        "This appendix is generated from selected_settings.json, validation_trials.csv, "
        "runtime_versions.json and cohort.csv. The final performance table uses results.csv only.", "",
        f"Documented tuning fits: {len(trials)}. Objective: LightGBM regression_l1; "
        "XGBoost reg:absoluteerror. Random seed: 42. Predictions are clipped at zero.", "",
        "| Horizon | Model | Selected trial | Locked parameters |", "| --- | --- | --- | --- |",
    ]
    for horizon, record in settings.items():
        for model, params in record["models"].items():
            subset = trials[(trials.horizon == horizon) & (trials.model == model)]
            chosen = subset.loc[subset.validation_macro_mae.idxmin()]
            appendix.append(f"| {horizon} | {model} | {int(chosen.trial_id)} | `{json.dumps(params, sort_keys=True)}` |")
    appendix += ["", "## Available inputs", ""]
    for horizon, record in settings.items():
        appendix.append(f"- {horizon}: {', '.join(record['features'])}.")
    appendix += ["", "## Runtime", "", f"Python {runtime['python']}.", ""]
    appendix += [f"- {name}: {value}" for name, value in runtime["packages"].items()]
    appendix += [
        "", "## Reproduce the workflow", "",
        "The executed experiment is locked. Do not delete its final-test markers or retune "
        "after inspecting its results. The following commands describe the original sequence; "
        "a fresh reproduction must be identified as a reproduction of already-published settings.", "",
        "```bash", "python download_energy_data.py", "python prepare_energy_data.py",
        "python run_multihorizon.py --stage tune --horizon all",
        "python run_multihorizon.py --stage final --horizon all",
        "python verify_lag_spot_check.py", "python generate_run_report.py", "python run_analysis_starter.py", "```", "",
        "Raw and prepared data files and large prediction exports are excluded from Git. "
        "The preparation manifest and final lock record their checksums.", "",
    ]
    appendix_text = "\n".join(appendix)
    (output_dir / "executed_settings_appendix.md").write_text(appendix_text)
    (output_dir / "report_sections.md").write_text(
        methods + "\n\n" + (output_dir / "headline_results.md").read_text() + "\n\n" + appendix_text
    )

    handover = [
        "# Modelling handover", "",
        "This handover reads only saved predictions, metrics and settings; it does not retrain or reopen the test experiment.", "",
        "- results.csv and table2.md: authoritative headline results and every model/horizon comparison.",
        "- predictions.csv: building_id, timestamp, horizon, model, actual_kwh, forecast_kwh.",
        "- predictions_wide.csv: baseline and both model forecasts on the same row for plotting.",
        "- per_building_metrics_with_type.csv and error_by_dwelling_type.csv: house/unit analysis.",
        "- feature_importance.csv: within-model importance; library scales are not directly comparable.",
        "- ../notebooks/predictions_starter.ipynb: starter analysis for Joseph and Regina.",
        "- validation_trials.csv, selected_settings.json and executed_settings_appendix.md: tried and executed settings.",
        "- ../docs/handover_notes.md: proposal changes, cohort limits and source assumptions.", "",
        "Choose an apartment case by customer ID independently of its error. The team has not supplied "
        "the ID for its specific apartment building. An illustrative unit must not be described as the team's own building.", "",
        "Do not infer causation from feature importance. LightGBM exports split-count importance; "
        "XGBoost exports its default importance. Normalise and interpret each model separately.", "",
    ]
    (output_dir / "HANDOVER.md").write_text("\n".join(handover))
    print(f"Report sections and typed building-error handover saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("multihorizon_outputs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    args = parser.parse_args()
    main(args.output_dir, args.data_dir)
