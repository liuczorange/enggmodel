# Metric definitions and a hand-worked check

For each household with n scored half-hour targets, MAE is the mean absolute forecast error and RMSE is the square root of the mean squared forecast error. NMAE divides MAE by the household's mean absolute observed consumption; it is undefined when that mean is zero. R² is reported only for nonconstant observations with more than one row.

Tiny verification example (illustrative arithmetic, not experimental results): observed consumption is [1, 2, 3] kWh and forecasts are [1, 4, 2] kWh. Absolute errors are [0, 2, 1], giving MAE = 3/3 = 1 kWh. Squared errors are [0, 4, 1], giving RMSE = sqrt(5/3) kWh. Mean observed consumption is 2 kWh, so NMAE = 1/2 = 0.5. `test_metrics_match_hand_calculation` checks these values.

Macro MAE is the arithmetic mean of the household MAEs. Macro RMSE is the arithmetic mean of household RMSEs, not the square root of the pooled squared errors. Pooled metrics combine all observations and are supplementary; pooled R² is not household-balanced. Equal household weighting prevents customers with more observations from receiving more weight; it does not make the absolute errors scale-free.

For each horizon, improvement = 100 × (baseline macro MAE − model macro MAE) / baseline macro MAE. A positive value means a reduction in error. Equality does not count as beating the baseline. If baseline error were zero, percentage improvement would be undefined and should not be interpreted.
