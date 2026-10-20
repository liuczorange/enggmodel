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
