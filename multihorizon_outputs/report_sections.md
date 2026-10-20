# Modelling methods

## Forecasting task

We predicted half-hourly household electricity consumption (`KWH_READING`, kWh) at 1 hour, 3 days and 1 week ahead. At target time T, consumption predictors had to be observed by T minus the horizon. Target calendar attributes were known in advance; realised target-time market price and system demand were excluded.

## Leakage-safe predictors

Lags were rebuilt by exact customer/timestamp lookup, avoiding row-shift errors across missing intervals. Candidate lags were 1 hour, 3 hours, 24 hours, 3 days, 1 week, 2 weeks and 3 weeks; only lags at least as old as the horizon were admitted. Calendar predictors were cyclic hour, weekday and month encodings plus a weekend indicator. Customer IDs were excluded from predictors.

## Models and tuning

For a reproducible laptop-scale experiment, a fixed random sample of 100 learning, 30 fine-tuning and 30 held-out test customers was drawn from the prepared customer-disjoint split. These caps were fixed before model selection. The selected customer IDs and random seeds are retained; conclusions apply to this cohort rather than the entire source population.

LightGBM and XGBoost were compared using identical inputs, splits and metrics. Both add successive trees to reduce the current loss; their different tree-building and regularisation implementations provide a practical robustness check. Absolute-error objectives were used.

For each library/horizon, three documented settings were fitted on learning households and selected by minimum macro MAE on different fine-tuning households. Settings were then locked, models refitted on learning plus fine-tuning households, and separate test households evaluated once. File locks guard against repeated evaluation or subsequent tuning in this experiment directory.

## Baselines and evaluation

Each horizon used its matching historical reading as the free guess. Headline MAE and RMSE were calculated per household, then averaged with equal household weights. Pooled metrics were supplementary. A model beat the baseline only if its macro MAE was strictly lower. Longer horizons generally lose recent-load information, but monotonic deterioration was assessed from results rather than assumed.

## Reproducibility

All attempted settings, selected settings, runtime versions, customer IDs and input hashes were saved. Negative forecasts were clipped to zero. Predictions were exported by customer, target time, horizon and model; Table 2 and headline performance numbers were generated only from `results.csv`.


## Source-data limitations

The publisher has already filled missing SGSC readings. The original imputation may have used later observations, so the experiment is a retrospective evaluation on a processed dataset. Learning and test households cover the same calendar year; this evaluates transfer to unseen households, not a chronological future-year holdout. It does not establish prospective performance on a live stream. The original SGSC/AEMO timestamp join is retained; interval-start/end and daylight-saving alignment remain upstream assumptions. Realised market fields are excluded from prediction. Macro averaging gives each household equal weight, but it does not normalise away differences in consumption scale.


# Headline results

Table 2 reports errors averaged across test customers, giving each household equal weight regardless of its number of scored observations. Each machine-learning model is compared with the free historical guess matched to its forecast distance.

At **1 hour**, **xgboost** was the stronger machine-learning model (macro MAE **0.243 kWh**). It beat the matching free guess (macro MAE **0.301 kWh**), a change of **19.1%**.

At **3 days**, **lightgbm** was the stronger machine-learning model (macro MAE **0.288 kWh**). It beat the matching free guess (macro MAE **0.376 kWh**), a change of **23.4%**.

At **1 week**, **lightgbm** was the stronger machine-learning model (macro MAE **0.301 kWh**). It beat the matching free guess (macro MAE **0.374 kWh**), a change of **19.5%**.

Across the three distances, the best machine-learning macro MAE values were 1 hour: 0.243 kWh, 3 days: 0.288 kWh, 1 week: 0.301 kWh. This sequence should be interpreted together with each distance's own free guess rather than against zero error.

## Table 2

| Forecast distance | Model | Macro MAE (kWh) | Macro RMSE (kWh) | Pooled R² | MAE improvement vs free guess (%) | Beats free guess |
| --- | --- | --- | --- | --- | --- | --- |
| 1h | naive_1h | 0.301 | 0.592 | 0.313 | 0.0 | — |
| 1h | lightgbm | 0.245 | 0.493 | 0.533 | 18.36 | True |
| 1h | xgboost | 0.243 | 0.48 | 0.553 | 19.126 | True |
| 3d | naive_3d | 0.376 | 0.697 | 0.074 | 0.0 | — |
| 3d | lightgbm | 0.288 | 0.564 | 0.392 | 23.359 | True |
| 3d | xgboost | 0.29 | 0.553 | 0.414 | 22.868 | True |
| 1w | naive_1w | 0.374 | 0.695 | 0.083 | 0.0 | — |
| 1w | lightgbm | 0.301 | 0.579 | 0.359 | 19.537 | True |
| 1w | xgboost | 0.304 | 0.57 | 0.378 | 18.651 | True |


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
