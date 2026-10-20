from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from multihorizon.data import add_exact_lags, feature_columns
from multihorizon.metrics import evaluate_by_building, metric_values
from multihorizon.pipeline import final_test, tune


def _write_fixture(directory: Path, customers=8, days=29):
    directory.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2013-01-01", periods=days * 48, freq="30min")
    frames = []
    rng = np.random.default_rng(5)
    for customer in range(customers):
        h = times.hour + times.minute / 60
        weekday = times.dayofweek < 5
        values = 2 + customer * .25 + 1.2 * np.sin(2 * np.pi * (h - 6) / 24)
        values += .6 * weekday + .2 * np.sin(2 * np.pi * np.arange(len(times)) / (48 * 7))
        values += rng.normal(0, .05, len(times))
        frame = pd.DataFrame({
            "DATETIME_30MIN": times, "CUSTOMER_ID": customer,
            "KWH_READING": np.clip(values, .05, None),
            "PRICE_AUD_MWH": 50 + 5 * np.sin(2 * np.pi * h / 24),
            "DEMAND_MW": 8000 + 300 * np.sin(2 * np.pi * h / 24),
            "HOUR": times.hour, "DAY_OF_WEEK": times.dayofweek,
            "IS_WEEKEND": (times.dayofweek >= 5).astype(int), "MONTH": times.month,
        })
        frame["KWH_LAG_1HR"] = frame["KWH_READING"].shift(2)
        frame["KWH_LAG_3DAYS"] = frame["KWH_READING"].shift(144)
        frame["KWH_LAG_1WEEK"] = frame["KWH_READING"].shift(336)
        frames.append(frame.dropna())
    data = pd.concat(frames, ignore_index=True)
    ids = sorted(data["CUSTOMER_ID"].unique())
    splits = {"learn": ids[:4], "finetune": ids[4:6], "test": ids[6:]}
    for name, selected in splits.items():
        data[data["CUSTOMER_ID"].isin(selected)].to_csv(directory / f"data_{name}.csv", index=False)


def test_metrics_match_hand_calculation():
    actual = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, 4.0, 2.0])
    result = metric_values(actual, prediction)
    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert result["nmae"] == pytest.approx(0.5)


def test_macro_mae_gives_households_equal_weight():
    frame = pd.DataFrame({"CUSTOMER_ID": ["small", "large", "large", "large"],
                          "KWH_READING": [1., 100., 100., 100.]})
    summary, per_household = evaluate_by_building(frame, [3., 100., 100., 100.], "example", "1h")
    assert summary["macro_mae"] == pytest.approx(1.)
    assert summary["pooled_mae"] == pytest.approx(.5)
    assert per_household.set_index("building_id").loc["small", "mae"] == 2


def test_exact_lag_does_not_jump_across_missing_interval():
    times = pd.date_range("2013-01-01", periods=8, freq="30min").delete(4)
    frame = pd.DataFrame({
        "CUSTOMER_ID": "A", "DATETIME_30MIN": times,
        "KWH_READING": np.arange(len(times), dtype=float),
        "KWH_LAG_1HR": np.nan, "KWH_LAG_3DAYS": np.nan, "KWH_LAG_1WEEK": np.nan,
    })
    rebuilt, _ = add_exact_lags(frame, "tiny")
    row = rebuilt[rebuilt["DATETIME_30MIN"] == pd.Timestamp("2013-01-01 03:00")].iloc[0]
    assert np.isnan(row["lag_1h"])


def test_horizon_feature_availability():
    assert "lag_1h" in feature_columns("1h")
    assert "lag_1h" not in feature_columns("3d")
    assert "lag_3d" in feature_columns("3d")
    assert feature_columns("1w")[:3] == ["lag_1w", "lag_2w", "lag_3w"]


def test_full_tune_and_locked_final(tmp_path, monkeypatch):
    from multihorizon import models
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    _write_fixture(data_dir)
    monkeypatch.setattr(models, "LIGHTGBM_GRID", [models.LIGHTGBM_GRID[0]])
    monkeypatch.setattr(models, "XGBOOST_GRID", [models.XGBOOST_GRID[0]])
    tune(data_dir, output_dir, "all")
    assert not (output_dir / "FINAL_TEST_COMPLETE.json").exists()
    manual = pd.read_csv(output_dir / "manual_lag_check_finetune.csv")
    assert np.allclose(manual[["difference_1h", "difference_3d", "difference_1w"]], 0)
    settings_path = output_dir / "selected_settings.json"
    original_settings = settings_path.read_text()
    settings_path.write_text(original_settings + "\n")
    with pytest.raises(RuntimeError, match="settings changed"):
        final_test(data_dir, output_dir, "all")
    assert not (output_dir / "FINAL_TEST_STARTED.json").exists()
    settings_path.write_text(original_settings)
    final_test(data_dir, output_dir, "all")
    assert (output_dir / "FINAL_TEST_STARTED.json").exists()
    results = pd.read_csv(output_dir / "results.csv")
    predictions = pd.read_csv(output_dir / "predictions.csv")
    predictions_wide = pd.read_csv(output_dir / "predictions_wide.csv")
    assert len(results) == 9
    assert set(results["horizon"]) == {"1h", "3d", "1w"}
    assert list(predictions.columns) == [
        "building_id", "timestamp", "horizon", "model", "actual_kwh", "forecast_kwh"
    ]
    assert {"baseline_prediction", "lightgbm_prediction", "xgboost_prediction"} <= set(predictions_wide)
    with pytest.raises(RuntimeError, match="locked"):
        final_test(data_dir, output_dir, "all")
    with pytest.raises(RuntimeError, match="locked"):
        tune(data_dir, output_dir, "all")
