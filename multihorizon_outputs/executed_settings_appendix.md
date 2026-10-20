# Appendix: executed model settings

This appendix is generated from selected_settings.json, validation_trials.csv, runtime_versions.json and cohort.csv. The final performance table uses results.csv only.

Documented tuning fits: 18. Objective: LightGBM regression_l1; XGBoost reg:absoluteerror. Random seed: 42. Predictions are clipped at zero.

| Horizon | Model | Selected trial | Locked parameters |
| --- | --- | --- | --- |
| 1h | lightgbm | 2 | `{"learning_rate": 0.04, "max_depth": 8, "min_child_samples": 40, "n_estimators": 350, "num_leaves": 31}` |
| 1h | xgboost | 2 | `{"learning_rate": 0.04, "max_depth": 7, "min_child_weight": 5, "n_estimators": 350}` |
| 1w | lightgbm | 2 | `{"learning_rate": 0.04, "max_depth": 8, "min_child_samples": 40, "n_estimators": 350, "num_leaves": 31}` |
| 1w | xgboost | 2 | `{"learning_rate": 0.04, "max_depth": 7, "min_child_weight": 5, "n_estimators": 350}` |
| 3d | lightgbm | 2 | `{"learning_rate": 0.04, "max_depth": 8, "min_child_samples": 40, "n_estimators": 350, "num_leaves": 31}` |
| 3d | xgboost | 2 | `{"learning_rate": 0.04, "max_depth": 7, "min_child_weight": 5, "n_estimators": 350}` |

## Available inputs

- 1h: lag_1h, lag_3h, lag_24h, lag_3d, lag_1w, lag_2w, lag_3w, hour_sin, hour_cos, day_sin, day_cos, is_weekend, month_sin, month_cos.
- 1w: lag_1w, lag_2w, lag_3w, hour_sin, hour_cos, day_sin, day_cos, is_weekend, month_sin, month_cos.
- 3d: lag_3d, lag_1w, lag_2w, lag_3w, hour_sin, hour_cos, day_sin, day_cos, is_weekend, month_sin, month_cos.

## Runtime

Python 3.14.7.

- joblib: 1.6.0
- lightgbm: 4.7.0
- numpy: 2.5.3
- pandas: 2.3.3
- scikit-learn: 1.9.1
- xgboost: 3.4.1

## Reproduce the workflow

The executed experiment is locked. Do not delete its final-test markers or retune after inspecting its results. The following commands describe the original sequence; a fresh reproduction must be identified as a reproduction of already-published settings.

```bash
python download_energy_data.py
python prepare_energy_data.py
python run_multihorizon.py --stage tune --horizon all
python run_multihorizon.py --stage final --horizon all
python verify_lag_spot_check.py
python generate_run_report.py
python run_analysis_starter.py
```

Raw and prepared data files and large prediction exports are excluded from Git. The preparation manifest and final lock record their checksums.
