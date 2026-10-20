from pathlib import Path
import pandas as pd

from generate_headline_results import main


def test_headline_numbers_come_from_results_file(tmp_path: Path):
    results = pd.DataFrame([
        {"horizon": "1h", "model": "naive_1h", "macro_mae": 2.0,
         "macro_rmse": 3.0, "pooled_r2": .5, "baseline_macro_mae": 2.0,
         "macro_mae_improvement_pct": 0.0, "beats_baseline": pd.NA},
        {"horizon": "1h", "model": "xgboost", "macro_mae": 1.5,
         "macro_rmse": 2.5, "pooled_r2": .7, "baseline_macro_mae": 2.0,
         "macro_mae_improvement_pct": 25.0, "beats_baseline": True},
        {"horizon": "1h", "model": "lightgbm", "macro_mae": 1.8,
         "macro_rmse": 2.8, "pooled_r2": .6, "baseline_macro_mae": 2.0,
         "macro_mae_improvement_pct": 10.0, "beats_baseline": True},
        {"horizon": "3d", "model": "naive_3d", "macro_mae": 1.0,
         "macro_rmse": 1.5, "pooled_r2": .4, "baseline_macro_mae": 1.0,
         "macro_mae_improvement_pct": 0.0, "beats_baseline": pd.NA},
        {"horizon": "3d", "model": "xgboost", "macro_mae": 1.2,
         "macro_rmse": 1.6, "pooled_r2": .3, "baseline_macro_mae": 1.0,
         "macro_mae_improvement_pct": -20.0, "beats_baseline": False},
        {"horizon": "3d", "model": "lightgbm", "macro_mae": 1.3,
         "macro_rmse": 1.7, "pooled_r2": .2, "baseline_macro_mae": 1.0,
         "macro_mae_improvement_pct": -30.0, "beats_baseline": False},
    ])
    source = tmp_path / "results.csv"
    output = tmp_path / "headline.md"
    results.to_csv(source, index=False)
    main(source, output)
    text = output.read_text()
    assert "1.500 kWh" in text
    assert "2.000 kWh" in text
    assert "25.0%" in text
    assert "did not beat" in text
    assert "Table 2" in text
