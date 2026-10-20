"""Configuration shared by the three forecasting horizons."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


RANDOM_SEED = 42
MAX_BUILDINGS_BY_SPLIT = {"learn": 100, "finetune": 30, "test": 30}


@dataclass(frozen=True)
class Horizon:
    name: str
    delta: timedelta
    baseline_feature: str


HORIZONS = {
    "1h": Horizon("1h", timedelta(hours=1), "lag_1h"),
    "3d": Horizon("3d", timedelta(days=3), "lag_3d"),
    "1w": Horizon("1w", timedelta(days=7), "lag_1w"),
}

# Exact timestamp lags. A horizon may only use lags at least as old as itself.
LAG_DELTAS = {
    "lag_1h": timedelta(hours=1),
    "lag_3h": timedelta(hours=3),
    "lag_24h": timedelta(days=1),
    "lag_3d": timedelta(days=3),
    "lag_1w": timedelta(days=7),
    "lag_2w": timedelta(days=14),
    "lag_3w": timedelta(days=21),
}

TIME_FEATURES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_weekend", "month_sin", "month_cos",
]

LIGHTGBM_GRID = [
    {"learning_rate": 0.05, "num_leaves": 15, "max_depth": 6, "n_estimators": 250, "min_child_samples": 30},
    {"learning_rate": 0.04, "num_leaves": 31, "max_depth": 8, "n_estimators": 350, "min_child_samples": 40},
    {"learning_rate": 0.08, "num_leaves": 15, "max_depth": 5, "n_estimators": 200, "min_child_samples": 50},
]

XGBOOST_GRID = [
    {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 250, "min_child_weight": 3},
    {"learning_rate": 0.04, "max_depth": 7, "n_estimators": 350, "min_child_weight": 5},
    {"learning_rate": 0.08, "max_depth": 4, "n_estimators": 200, "min_child_weight": 5},
]
