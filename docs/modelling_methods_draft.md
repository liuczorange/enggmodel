# Modelling methods draft

> Final prose should be shortened to the allocated 1.25 pages. Replace bracketed cross-references only after the locked test run. No numerical result should be typed manually; Table 2 is generated from `results.csv`.

## Forecasting task

We predicted half-hourly household electricity consumption (`KWH_READING`, kWh) at 1 hour, 3 days and 1 week ahead. At target time T, consumption predictors had to be observed by T minus the horizon. Target calendar attributes were known in advance; realised target-time market price and system demand were excluded.

## Leakage-safe predictors

Lags were rebuilt by exact customer/timestamp lookup, avoiding row-shift errors across missing intervals. Candidate lags were 1 hour, 3 hours, 24 hours, 3 days, 1 week, 2 weeks and 3 weeks; only lags at least as old as the horizon were admitted. Calendar predictors were cyclic hour, weekday and month encodings plus a weekend indicator. Customer IDs were excluded from predictors.

## Models and tuning

LightGBM and XGBoost were compared using identical inputs, splits and metrics. Both add successive trees to reduce the current loss; their different tree-building and regularisation implementations provide a practical robustness check. Absolute-error objectives were used.

For each library/horizon, three documented settings were fitted on learning households and selected by minimum macro MAE on different fine-tuning households. Settings were then locked, models refitted on learning plus fine-tuning households, and separate test households evaluated once. File locks guard against repeated evaluation or subsequent tuning in this experiment directory.

## Baselines and evaluation

Each horizon used its matching historical reading as the free guess. Headline MAE and RMSE were calculated per household, then averaged with equal household weights. Pooled metrics were supplementary. A model beat the baseline only if its macro MAE was strictly lower. Longer horizons generally lose recent-load information, but monotonic deterioration was assessed from results rather than assumed.

## Reproducibility

All attempted settings, selected settings, runtime versions, customer IDs and input hashes were saved. Negative forecasts were clipped to zero. Predictions were exported by customer, target time, horizon and model; Table 2 and headline performance numbers were generated only from `results.csv`.
