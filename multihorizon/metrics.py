"""Pooled and building-balanced forecast metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import BUILDING, TARGET


def metric_values(actual, prediction) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mae = mean_absolute_error(actual, prediction)
    rmse = mean_squared_error(actual, prediction) ** 0.5
    denominator = np.mean(np.abs(actual))
    return {
        "mae": float(mae), "rmse": float(rmse),
        "r2": float(r2_score(actual, prediction)) if len(actual) > 1 and np.var(actual) > 0 else np.nan,
        "nmae": float(mae / denominator) if denominator > 0 else np.nan,
    }


def evaluate_by_building(frame: pd.DataFrame, prediction, model: str, horizon: str) -> tuple[dict, pd.DataFrame]:
    scored = frame[[BUILDING, TARGET]].copy()
    scored["prediction"] = np.asarray(prediction)
    rows = []
    for building, group in scored.groupby(BUILDING, sort=False):
        rows.append({
            "horizon": horizon, "model": model, "building_id": building,
            "n_rows": len(group), **metric_values(group[TARGET], group["prediction"]),
        })
    per_building = pd.DataFrame(rows)
    pooled = metric_values(scored[TARGET], scored["prediction"])
    summary = {
        "horizon": horizon, "model": model,
        "pooled_mae": pooled["mae"], "pooled_rmse": pooled["rmse"],
        "pooled_r2": pooled["r2"], "pooled_nmae": pooled["nmae"],
        "macro_mae": per_building["mae"].mean(),
        "macro_rmse": per_building["rmse"].mean(),
        "n_buildings": per_building["building_id"].nunique(),
        "n_rows": len(scored),
    }
    return summary, per_building
