from pathlib import Path
import pandas as pd

from generate_table2 import main


def test_table2_is_generated_only_from_results(tmp_path: Path):
    results = pd.DataFrame([{
        "horizon": "1h", "model": "xgboost", "macro_mae": 1.23456,
        "macro_rmse": 2.34567, "pooled_r2": .87654,
        "macro_mae_improvement_pct": 12.3456, "beats_baseline": True,
    }])
    source = tmp_path / "results.csv"
    output = tmp_path / "table2.md"
    results.to_csv(source, index=False)
    main(source, output)
    text = output.read_text()
    assert "1.235" in text
    assert "12.346" in text
    assert (tmp_path / "table2.csv").exists()
