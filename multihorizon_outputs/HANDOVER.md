# Modelling handover

This handover reads only saved predictions, metrics and settings; it does not retrain or reopen the test experiment.

- results.csv and table2.md: authoritative headline results and every model/horizon comparison.
- predictions.csv: building_id, timestamp, horizon, model, actual_kwh, forecast_kwh.
- predictions_wide.csv: baseline and both model forecasts on the same row for plotting.
- per_building_metrics_with_type.csv and error_by_dwelling_type.csv: house/unit analysis.
- feature_importance.csv: within-model importance; library scales are not directly comparable.
- ../notebooks/predictions_starter.ipynb: starter analysis for Joseph and Regina.
- validation_trials.csv, selected_settings.json and executed_settings_appendix.md: tried and executed settings.
- ../docs/handover_notes.md: proposal changes, cohort limits and source assumptions.

Choose an apartment case by customer ID independently of its error. The team has not supplied the ID for its specific apartment building. An illustrative unit must not be described as the team's own building.

Do not infer causation from feature importance. LightGBM exports split-count importance; XGBoost exports its default importance. Normalise and interpret each model separately.
