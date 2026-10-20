"""Two-stage pipeline: tune without test data, then evaluate the test split once."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd

from .config import HORIZONS, MAX_BUILDINGS_BY_SPLIT, RANDOM_SEED
from .data import (
    BUILDING, TARGET, TIMESTAMP, add_exact_lags, assert_disjoint_buildings,
    load_split, prepare_horizon,
)
from .metrics import evaluate_by_building
from .models import fit_final_model, importance_frame, tune_model


def _paths(data_dir: Path) -> dict[str, Path]:
    return {
        "learn": data_dir / "data_learn.csv",
        "finetune": data_dir / "data_finetune.csv",
        "test": data_dir / "data_test.csv",
    }


def _horizons(selection: str) -> list[str]:
    return list(HORIZONS) if selection == "all" else [selection]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_selected(path: Path, split_name: str) -> pd.DataFrame:
    seed_offset = {"learn": 0, "finetune": 1, "test": 2}[split_name]
    print(f"Loading {split_name}: fixed cap {MAX_BUILDINGS_BY_SPLIT[split_name]} customers", flush=True)
    data = load_split(
        path, max_buildings=MAX_BUILDINGS_BY_SPLIT[split_name],
        random_seed=RANDOM_SEED + seed_offset,
    )
    print(f"Loaded {split_name}: {data[BUILDING].nunique()} customers, {len(data):,} rows", flush=True)
    return data


def _cohort_rows(data: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "split": split_name,
        "building_id": sorted(data[BUILDING].unique()),
    })


def _manual_lag_check(data: pd.DataFrame) -> pd.DataFrame:
    """Human-readable spot check for one fine-tuning building."""
    building = sorted(data[BUILDING].unique())[0]
    columns = [
        BUILDING, TIMESTAMP, TARGET,
        "KWH_LAG_1HR", "lag_1h", "KWH_LAG_3DAYS", "lag_3d",
        "KWH_LAG_1WEEK", "lag_1w",
    ]
    check = data[data[BUILDING] == building][columns].dropna().head(10).copy()
    check["difference_1h"] = check["lag_1h"] - check["KWH_LAG_1HR"]
    check["difference_3d"] = check["lag_3d"] - check["KWH_LAG_3DAYS"]
    check["difference_1w"] = check["lag_1w"] - check["KWH_LAG_1WEEK"]
    return check


def tune(data_dir: Path, output_dir: Path, horizon_selection: str) -> None:
    """Choose model settings from learn/finetune only; never reads data_test.csv."""
    if any((output_dir / marker).exists() for marker in ("FINAL_TEST_STARTED.json", "FINAL_TEST_COMPLETE.json")):
        raise RuntimeError("Final test is locked; do not retune this experiment after opening test data")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(data_dir)
    learn, audit_learn = add_exact_lags(_load_selected(paths["learn"], "learn"), "learn")
    finetune, audit_finetune = add_exact_lags(_load_selected(paths["finetune"], "finetune"), "finetune")
    overlap = set(learn[BUILDING]) & set(finetune[BUILDING])
    if overlap:
        raise ValueError(f"Building leakage between learn and finetune: {sorted(overlap)[:5]}")

    all_trials = []
    baseline_summaries = []
    selected = {}
    for horizon in _horizons(horizon_selection):
        learn_h, features = prepare_horizon(learn, horizon)
        finetune_h, tune_features = prepare_horizon(finetune, horizon)
        if features != tune_features:
            raise AssertionError("Feature schema differs between learn and finetune")
        baseline = np.clip(finetune_h[HORIZONS[horizon].baseline_feature].to_numpy(), 0, None)
        baseline_summary, _ = evaluate_by_building(finetune_h, baseline, f"naive_{horizon}", horizon)
        baseline_summaries.append(baseline_summary)
        print(f"VALIDATION {horizon} free guess: macro MAE={baseline_summary['macro_mae']:.6f}", flush=True)
        selected[horizon] = {"features": features, "models": {}}
        for model_name in ("lightgbm", "xgboost"):
            params, trials = tune_model(model_name, horizon, learn_h, finetune_h, features)
            selected[horizon]["models"][model_name] = params
            all_trials.append(trials)
            # Checkpoint completed searches, without ever reading test performance.
            pd.concat(all_trials, ignore_index=True).to_csv(output_dir / "validation_trials.csv", index=False)

    pd.concat(all_trials, ignore_index=True).to_csv(output_dir / "validation_trials.csv", index=False)
    pd.DataFrame(baseline_summaries).to_csv(output_dir / "validation_baselines.csv", index=False)
    pd.concat([audit_learn, audit_finetune], ignore_index=True).to_csv(
        output_dir / "lag_audit_tuning.csv", index=False
    )
    pd.concat([
        _cohort_rows(learn, "learn"), _cohort_rows(finetune, "finetune")
    ], ignore_index=True).to_csv(output_dir / "cohort.csv", index=False)
    _manual_lag_check(finetune).to_csv(output_dir / "manual_lag_check_finetune.csv", index=False)
    _write_json(output_dir / "selected_settings.json", selected)
    _write_json(output_dir / "runtime_versions.json", {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "joblib")},
        "cohort_caps": MAX_BUILDINGS_BY_SPLIT,
        "cohort_seeds": {"learn": RANDOM_SEED, "finetune": RANDOM_SEED+1, "test": RANDOM_SEED+2},
    })
    _write_json(output_dir / "tuning_complete.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": _horizons(horizon_selection),
        "note": "No test file was read during this stage.",
        "input_sha256": {name: _sha256(paths[name]) for name in ("learn", "finetune")},
        "settings_sha256": _sha256(output_dir / "selected_settings.json"),
        "runtime_sha256": _sha256(output_dir / "runtime_versions.json"),
    })


def final_test(data_dir: Path, output_dir: Path, horizon_selection: str) -> None:
    """Refit selected settings, then generate test predictions exactly once."""
    marker = output_dir / "FINAL_TEST_COMPLETE.json"
    started_marker = output_dir / "FINAL_TEST_STARTED.json"
    if marker.exists() or started_marker.exists():
        raise RuntimeError(
            "Final test is locked because a final-test marker exists. "
            "Do not repeatedly inspect test performance."
        )
    settings_path = output_dir / "selected_settings.json"
    if not settings_path.exists():
        raise FileNotFoundError("Run the tune stage before the final stage.")
    settings = json.loads(settings_path.read_text())
    tuning_record = json.loads((output_dir / "tuning_complete.json").read_text())
    if _sha256(settings_path) != tuning_record["settings_sha256"]:
        raise RuntimeError("Selected settings changed after tuning; final test has not been opened")
    runtime = json.loads((output_dir / "runtime_versions.json").read_text())
    if runtime["cohort_caps"] != MAX_BUILDINGS_BY_SPLIT:
        raise RuntimeError("Cohort caps changed after tuning; final test has not been opened")
    requested = _horizons(horizon_selection)
    missing_settings = [h for h in requested if h not in settings]
    if missing_settings:
        raise ValueError(f"No tuned settings for horizons: {missing_settings}")

    paths = _paths(data_dir)
    for name in ("learn", "finetune"):
        if _sha256(paths[name]) != tuning_record["input_sha256"][name]:
            raise RuntimeError(f"{name} data changed after tuning; rerun tuning before final test")
    # From this point onward the test has been opened. The marker remains even if a later step fails.
    _write_json(started_marker, {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": requested,
        "warning": "Test access started; do not delete this marker to rerun after seeing test data.",
    })
    # Test is deliberately loaded only here, after every setting has been fixed.
    learn, audit_learn = add_exact_lags(_load_selected(paths["learn"], "learn"), "learn")
    finetune, audit_finetune = add_exact_lags(_load_selected(paths["finetune"], "finetune"), "finetune")
    test, audit_test = add_exact_lags(_load_selected(paths["test"], "test"), "test")
    assert_disjoint_buildings(learn, finetune, test)
    combined = pd.concat([learn, finetune], ignore_index=True)

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    summaries, building_metrics, prediction_frames, importances = [], [], [], []
    for horizon in requested:
        combined_h, features = prepare_horizon(combined, horizon)
        test_h, test_features = prepare_horizon(test, horizon)
        if features != test_features or features != settings[horizon]["features"]:
            raise AssertionError("Locked feature schema does not match final data")

        baseline_name = f"naive_{horizon}"
        baseline = np.clip(test_h[HORIZONS[horizon].baseline_feature].to_numpy(), 0, None)
        baseline_summary, baseline_buildings = evaluate_by_building(test_h, baseline, baseline_name, horizon)
        summaries.append(baseline_summary); building_metrics.append(baseline_buildings)
        predictions = test_h[[BUILDING, TIMESTAMP, TARGET]].rename(columns={
            BUILDING: "building_id", TIMESTAMP: "timestamp", TARGET: "actual_kwh",
        })
        predictions.insert(2, "horizon", horizon)
        predictions["baseline_prediction"] = baseline

        for model_name in ("lightgbm", "xgboost"):
            params = settings[horizon]["models"][model_name]
            model = fit_final_model(
                model_name, params, combined_h, features,
                model_dir / f"{model_name}_{horizon}.joblib",
            )
            forecast = np.clip(model.predict(test_h[features]), 0, None)
            summary, per_building = evaluate_by_building(test_h, forecast, model_name, horizon)
            print(f"FINAL {horizon} {model_name}: macro MAE={summary['macro_mae']:.6f}, free guess={baseline_summary['macro_mae']:.6f}", flush=True)
            summaries.append(summary); building_metrics.append(per_building)
            predictions[f"{model_name}_prediction"] = forecast
            importances.append(importance_frame(model, features, model_name, horizon))
        prediction_frames.append(predictions)

    summary = pd.DataFrame(summaries)
    baseline_mae = summary[summary["model"].str.startswith("naive_")].set_index("horizon")["macro_mae"]
    summary["baseline_macro_mae"] = summary["horizon"].map(baseline_mae)
    summary["macro_mae_improvement_pct"] = (
        100 * (summary["baseline_macro_mae"] - summary["macro_mae"]) / summary["baseline_macro_mae"]
    )
    summary["beats_baseline"] = (
        summary["macro_mae"] < summary["baseline_macro_mae"]
    ).astype("boolean")
    summary.loc[summary["model"].str.startswith("naive_"), "beats_baseline"] = pd.NA
    summary.to_csv(output_dir / "results.csv", index=False)
    pd.concat(building_metrics, ignore_index=True).to_csv(output_dir / "per_building_metrics.csv", index=False)
    predictions_wide = pd.concat(prediction_frames, ignore_index=True)
    predictions_wide.to_csv(output_dir / "predictions_wide.csv", index=False)
    prediction_columns = {
        "baseline_prediction": "baseline",
        "lightgbm_prediction": "lightgbm",
        "xgboost_prediction": "xgboost",
    }
    predictions_long = predictions_wide.melt(
        id_vars=["building_id", "timestamp", "horizon", "actual_kwh"],
        value_vars=list(prediction_columns), var_name="prediction_source",
        value_name="forecast_kwh",
    )
    predictions_long["model"] = predictions_long["prediction_source"].map(prediction_columns)
    predictions_long = predictions_long[
        ["building_id", "timestamp", "horizon", "model", "actual_kwh", "forecast_kwh"]
    ]
    predictions_long.to_csv(output_dir / "predictions.csv", index=False)
    pd.concat(importances, ignore_index=True).to_csv(output_dir / "feature_importance.csv", index=False)
    pd.concat([audit_learn, audit_finetune, audit_test], ignore_index=True).to_csv(
        output_dir / "lag_audit_final.csv", index=False
    )
    pd.concat([
        _cohort_rows(learn, "learn"), _cohort_rows(finetune, "finetune"),
        _cohort_rows(test, "test"),
    ], ignore_index=True).to_csv(output_dir / "cohort.csv", index=False)
    _write_json(marker, {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": requested,
        "test_buildings": int(test[BUILDING].nunique()),
        "test_file": str(paths["test"]),
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "settings_sha256": _sha256(settings_path),
        "warning": "Started and completed markers prevent accidental repeated final evaluation.",
    })
