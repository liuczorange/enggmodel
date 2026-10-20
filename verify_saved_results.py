#!/usr/bin/env python3
"""Check exports against saved forecasts without retraining or reopening source test data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from download_energy_data import sha256
from multihorizon.metrics import evaluate_by_building


PREDICTION_COLUMNS = {"baseline": "baseline_prediction", "lightgbm": "lightgbm_prediction",
                      "xgboost": "xgboost_prediction"}
KEY = ["building_id", "timestamp", "horizon"]


def verify(output_dir: Path) -> dict:
    final = json.loads((output_dir / "FINAL_TEST_COMPLETE.json").read_text())
    tuning = json.loads((output_dir / "tuning_complete.json").read_text())
    assert sha256(output_dir / "selected_settings.json") == final["settings_sha256"] == tuning["settings_sha256"]
    results = pd.read_csv(output_dir / "results.csv")
    cohort = pd.read_csv(output_dir / "cohort.csv", dtype={"building_id": "string"})
    groups = {name: set(group.building_id) for name, group in cohort.groupby("split")}
    assert not (groups["learn"] & groups["finetune"] or groups["learn"] & groups["test"] or groups["finetune"] & groups["test"])
    # XGBoost returned float32 forecasts. CSV writes those with float32 precision;
    # restore that dtype before comparing to the original, pre-export scores.
    wide = pd.read_csv(output_dir / "predictions_wide.csv",
                       dtype={"building_id": "string", "xgboost_prediction": "float32"},
                       parse_dates=["timestamp"])
    assert not wide.duplicated(KEY).any()
    assert set(wide.building_id) == groups["test"]
    assert np.isfinite(wide[["actual_kwh", *PREDICTION_COLUMNS.values()]].to_numpy()).all()
    assert wide[list(PREDICTION_COLUMNS.values())].ge(0).all().all()
    assert len(results) == 9
    expected_per_building = []
    for horizon, frame in wide.groupby("horizon"):
        scoring = frame.rename(columns={"building_id": "CUSTOMER_ID", "actual_kwh": "KWH_READING"})
        baseline_mae = results.loc[(results.horizon == horizon) & (results.model == f"naive_{horizon}"), "macro_mae"].item()
        for model, column in PREDICTION_COLUMNS.items():
            model_label = f"naive_{horizon}" if model == "baseline" else model
            computed, per_building = evaluate_by_building(scoring, frame[column].to_numpy(), model_label, horizon)
            stored = results[(results.horizon == horizon) & (results.model == model_label)].iloc[0]
            for metric in ["pooled_mae", "pooled_rmse", "pooled_r2", "pooled_nmae", "macro_mae", "macro_rmse", "n_buildings", "n_rows"]:
                assert np.isclose(computed[metric], stored[metric], rtol=1e-9, atol=1e-10, equal_nan=True), (horizon, model_label, metric)
            assert np.isclose(stored.baseline_macro_mae, baseline_mae)
            assert np.isclose(stored.macro_mae_improvement_pct, 100*(baseline_mae-stored.macro_mae)/baseline_mae)
            if model != "baseline":
                assert bool(stored.beats_baseline) == (stored.macro_mae < baseline_mae)
            expected_per_building.append(per_building)
    columns = ["horizon", "model", "building_id", "n_rows", "mae", "rmse", "r2", "nmae"]
    stored_per = pd.read_csv(output_dir / "per_building_metrics.csv", dtype={"building_id": "string"})
    expected_per = pd.concat(expected_per_building, ignore_index=True)
    sort = ["horizon", "model", "building_id"]
    pd.testing.assert_frame_equal(
        stored_per[columns].sort_values(sort).reset_index(drop=True),
        expected_per[columns].sort_values(sort).reset_index(drop=True),
        check_dtype=False, rtol=1e-9, atol=1e-10,
    )
    lookup = wide.set_index(KEY)
    long_rows = 0
    long_counts = {}
    for chunk in pd.read_csv(output_dir / "predictions.csv", dtype={"building_id": "string"},
                             parse_dates=["timestamp"], chunksize=250_000):
        assert list(chunk.columns) == [*KEY, "model", "actual_kwh", "forecast_kwh"]
        for model, group in chunk.groupby("model"):
            assert model in PREDICTION_COLUMNS
            selected = lookup.reindex(pd.MultiIndex.from_frame(group[KEY]))
            assert np.allclose(group.actual_kwh.to_numpy(), selected.actual_kwh.to_numpy(), rtol=1e-9, atol=1e-10)
            assert np.allclose(group.forecast_kwh.to_numpy(), selected[PREDICTION_COLUMNS[model]].to_numpy(), rtol=1e-9, atol=1e-10)
            long_counts[model] = long_counts.get(model, 0) + len(group)
        long_rows += len(chunk)
    assert long_rows == len(wide) * 3
    assert long_counts == {model: len(wide) for model in PREDICTION_COLUMNS}
    summary = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "results_rows": len(results), "wide_prediction_rows": len(wide),
        "long_prediction_rows": long_rows, "tested_households": len(groups["test"]),
        "all_checks_passed": True,
        "note": "Metrics recomputed from saved forecasts only; no source test file or fitted model was opened.",
        "prediction_dtype_note": "XGBoost's wide-column forecasts are restored as float32 to match their original model output; the canonical long file stores the full promoted values.",
        "artifact_sha256": {name: sha256(output_dir / name) for name in [
            "results.csv", "per_building_metrics.csv", "predictions_wide.csv", "predictions.csv", "selected_settings.json",
        ]},
    }
    (output_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "artifact_sha256"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("multihorizon_outputs"))
    args = parser.parse_args()
    verify(args.output_dir)
