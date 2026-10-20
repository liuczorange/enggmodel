"""Lightweight validation tuning for LightGBM and XGBoost."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from .config import LIGHTGBM_GRID, RANDOM_SEED, XGBOOST_GRID
from .data import TARGET
from .metrics import evaluate_by_building


def _estimator(model_name: str, params: dict):
    if model_name == "lightgbm":
        return LGBMRegressor(
            **params, objective="regression_l1", random_state=RANDOM_SEED,
            n_jobs=-1, verbosity=-1,
        )
    if model_name == "xgboost":
        return XGBRegressor(
            **params, objective="reg:absoluteerror", random_state=RANDOM_SEED,
            n_jobs=-1, verbosity=0,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def tune_model(model_name: str, horizon: str, learn: pd.DataFrame,
               finetune: pd.DataFrame, features: list[str]):
    grid = LIGHTGBM_GRID if model_name == "lightgbm" else XGBOOST_GRID
    trials = []
    best = None
    for trial_id, params in enumerate(grid, start=1):
        started = perf_counter()
        print(f"TUNE {horizon} {model_name} trial {trial_id}/{len(grid)}: fitting {len(learn):,} rows", flush=True)
        model = _estimator(model_name, params)
        model.fit(learn[features], learn[TARGET])
        prediction = np.clip(model.predict(finetune[features]), 0, None)
        summary, _ = evaluate_by_building(finetune, prediction, model_name, horizon)
        row = {
            "horizon": horizon, "model": model_name, "trial_id": trial_id,
            "parameters": json.dumps(params, sort_keys=True),
            "validation_macro_mae": summary["macro_mae"],
            "validation_macro_rmse": summary["macro_rmse"],
            "elapsed_seconds": perf_counter() - started,
        }
        print(f"TUNE {horizon} {model_name} trial {trial_id}: macro MAE={summary['macro_mae']:.6f}, {row['elapsed_seconds']:.1f}s", flush=True)
        trials.append(row)
        if best is None or row["validation_macro_mae"] < best[0]:
            best = (row["validation_macro_mae"], params.copy())
    trials_frame = pd.DataFrame(trials)
    trials_frame["rank"] = trials_frame["validation_macro_mae"].rank(method="min").astype(int)
    trials_frame["selected"] = trials_frame["rank"] == 1
    assert best is not None
    return best[1], trials_frame


def fit_final_model(model_name: str, params: dict, combined_train: pd.DataFrame,
                    features: list[str], model_path: Path):
    started = perf_counter()
    print(f"FINAL fitting {model_path.stem}: {len(combined_train):,} training rows", flush=True)
    model = _estimator(model_name, params)
    model.fit(combined_train[features], combined_train[TARGET])
    joblib.dump({"model": model, "features": features, "parameters": params}, model_path)
    print(f"FINAL saved {model_path.name}: {perf_counter()-started:.1f}s", flush=True)
    return model


def importance_frame(model, features: list[str], model_name: str, horizon: str) -> pd.DataFrame:
    values = getattr(model, "feature_importances_", np.zeros(len(features)))
    return pd.DataFrame({
        "horizon": horizon, "model": model_name,
        "feature": features, "importance": values,
    }).sort_values("importance", ascending=False)
