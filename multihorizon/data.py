"""Read teammate outputs and rebuild every lag by exact timestamp."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import HORIZONS, LAG_DELTAS, TIME_FEATURES

TIMESTAMP = "DATETIME_30MIN"
BUILDING = "CUSTOMER_ID"
TARGET = "KWH_READING"
REQUIRED = {
    TIMESTAMP, BUILDING, TARGET, "PRICE_AUD_MWH", "DEMAND_MW",
    "HOUR", "DAY_OF_WEEK", "IS_WEEKEND", "MONTH",
    "KWH_LAG_1HR", "KWH_LAG_3DAYS", "KWH_LAG_1WEEK",
}
PROVIDED_LAGS = {
    "lag_1h": "KWH_LAG_1HR",
    "lag_3d": "KWH_LAG_3DAYS",
    "lag_1w": "KWH_LAG_1WEEK",
}


def load_split(path: Path, max_buildings: int | None = None, random_seed: int = 42) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = sorted(REQUIRED - set(header.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    selected = None
    if max_buildings is not None:
        available: set[str] = set()
        for chunk in pd.read_csv(path, usecols=[BUILDING], dtype={BUILDING: "string"}, chunksize=1_000_000):
            available.update(chunk[BUILDING].dropna().astype(str).unique())
        ordered = np.array(sorted(available), dtype=object)
        if len(ordered) > max_buildings:
            rng = np.random.default_rng(random_seed)
            selected = set(rng.choice(ordered, size=max_buildings, replace=False).tolist())
        else:
            selected = set(ordered.tolist())

    chunks = []
    for chunk in pd.read_csv(path, dtype={BUILDING: "string"}, chunksize=500_000, low_memory=False):
        if selected is not None:
            chunk = chunk[chunk[BUILDING].astype(str).isin(selected)]
        if len(chunk):
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No rows selected from {path}")
    data = pd.concat(chunks, ignore_index=True)
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], errors="coerce")
    data[BUILDING] = data[BUILDING].astype(str)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    if data.duplicated([BUILDING, TIMESTAMP]).any():
        raise ValueError(f"{path} contains duplicate customer/timestamp rows")
    if data[TIMESTAMP].isna().any():
        raise ValueError(f"{path} contains invalid timestamps")
    return data.sort_values([BUILDING, TIMESTAMP]).reset_index(drop=True)


def add_exact_lags(data: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use timestamp lookup, never row position, so gaps cannot create false lags."""
    data = data.copy()
    lookup_series = data.set_index([BUILDING, TIMESTAMP])[TARGET]
    audit_rows = []
    for feature, delta in LAG_DELTAS.items():
        lookup_index = pd.MultiIndex.from_arrays(
            [data[BUILDING], data[TIMESTAMP] - delta], names=[BUILDING, TIMESTAMP]
        )
        rebuilt = lookup_series.reindex(lookup_index).to_numpy()
        data[feature] = rebuilt
        provided_name = PROVIDED_LAGS.get(feature)
        if provided_name:
            provided = pd.to_numeric(data[provided_name], errors="coerce").to_numpy()
            comparable = np.isfinite(rebuilt) & np.isfinite(provided)
            match = np.isclose(rebuilt[comparable], provided[comparable], rtol=1e-5, atol=1e-6)
            audit_rows.append({
                "split": split_name, "feature": feature,
                "comparable_rows": int(comparable.sum()),
                "matching_rows": int(match.sum()),
                "match_rate": float(match.mean()) if len(match) else np.nan,
                "max_absolute_difference": float(np.max(np.abs(rebuilt[comparable] - provided[comparable])))
                if comparable.any() else np.nan,
            })

    timestamp = data[TIMESTAMP]
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    day = timestamp.dt.dayofweek
    month = timestamp.dt.month - 1
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_sin"] = np.sin(2 * np.pi * day / 7)
    data["day_cos"] = np.cos(2 * np.pi * day / 7)
    data["is_weekend"] = (day >= 5).astype(np.int8)
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)
    return data, pd.DataFrame(audit_rows)


def feature_columns(horizon_name: str) -> list[str]:
    horizon = HORIZONS[horizon_name]
    safe_lags = [name for name, delta in LAG_DELTAS.items() if delta >= horizon.delta]
    return safe_lags + TIME_FEATURES


def prepare_horizon(data: pd.DataFrame, horizon_name: str) -> tuple[pd.DataFrame, list[str]]:
    columns = feature_columns(horizon_name)
    required = [BUILDING, TIMESTAMP, TARGET, *columns]
    clean = data[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    clean = clean[clean[TARGET] >= 0]
    if clean.empty:
        raise ValueError(f"No valid rows remain for horizon {horizon_name}")
    return clean, columns


def assert_disjoint_buildings(learn: pd.DataFrame, finetune: pd.DataFrame, test: pd.DataFrame) -> None:
    groups = {
        "learn": set(learn[BUILDING]),
        "finetune": set(finetune[BUILDING]),
        "test": set(test[BUILDING]),
    }
    pairs = [("learn", "finetune"), ("learn", "test"), ("finetune", "test")]
    for left, right in pairs:
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f"Building leakage between {left} and {right}: {sorted(overlap)[:5]}")
