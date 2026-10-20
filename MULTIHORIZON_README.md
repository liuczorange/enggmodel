# Multi-horizon SGSC electricity forecasting

> Published repository: the completed experiment is included and locked. Follow the root README to restore its compressed predictions or reproduce into separate `data/reproduction_cleaned/` and `reproduction_outputs/` directories. The default commands below describe the original local execution and must not overwrite the published experiment.

This is the modelling component that consumes the three CSV files created by the teammate repository [`eaych/energy-building-data-processing`](https://github.com/eaych/energy-building-data-processing).

It predicts **electricity consumption in kWh**, not the electricity bill. `PRICE_AUD_MWH` is a market-price field, but the target is `KWH_READING`.

## Download and prepare real data

The source is the public [SGSC dataset, version 1](https://data.mendeley.com/datasets/zm4f727vvr/1), licensed CC BY 4.0, together with AEMO NSW price-and-demand records for all 12 months of 2013. The interval file is approximately 990 MB. Allow several GB of local disk space for the prepared CSV files.

```bash
python download_energy_data.py
python prepare_energy_data.py
```

The preparation script uses the teammate's default 1,500-customer sample (sampling seed 123), the same modelling columns and 70/15/15 customer proportions. It fixes the split seed at 42, builds timestamp-exact lags, and processes one customer at a time instead of melting the entire table. It verifies the source's official SHA-256 checksum, requires all 17,520 AEMO settlements, and refuses to overwrite frozen outputs.

Outputs in `data/cleaned/` are `data_learn.csv`, `data_finetune.csv`, `data_test.csv`, `customer_metadata.csv` and `preparation_manifest.json`. The manifest records IDs, file hashes, row counts and preparation differences. The metadata contains the published house/unit assignments for Joseph's building-type figure; join it on `building_id`, and do not use its full-year energy statistics as model predictors.

The publisher has already filled missing source readings. The original filling procedure may have used later observations; consequently, these are retrospective tests on a published processed dataset, not proof of performance on a live, unimputed stream. The teammate's exact SGSC/AEMO timestamp join is retained; interval-label and daylight-saving conventions should be confirmed with the data owner. Realised market columns are not used by the models.

## One horizon dial

The same program supports three distances:

```bash
python run_multihorizon.py --stage tune --horizon 1h
python run_multihorizon.py --stage tune --horizon 3d
python run_multihorizon.py --stage tune --horizon 1w
```

For the official experiment, run all three together so the final test is opened only once:

```bash
python run_multihorizon.py --stage tune --horizon all \
  --data-dir /path/to/energy-building-data-processing/data/cleaned

python run_multihorizon.py --stage final --horizon all \
  --data-dir /path/to/energy-building-data-processing/data/cleaned
```

The tune stage reads only `data_learn.csv` and `data_finetune.csv`. It records every LightGBM and XGBoost setting tried in `validation_trials.csv`. The final stage locks its settings, refits on learn + finetune, reads `data_test.csv`, evaluates it once, and creates `FINAL_TEST_COMPLETE.json`. A second final run is refused.

The teammate preprocessing defaults to 1,500 customers (about 26 million half-hour rows). For a practical semester-project run, `multihorizon/config.py` deterministically selects at most 100 learn, 30 fine-tune and 30 test customers. The exact IDs are saved to `cohort.csv`. These caps are configuration constants rather than model-tuning choices and can be raised on a larger machine.

## Leakage rules

At target time `T`:

- the 1-hour model can use readings from `T-1h` or earlier;
- the 3-day model can use readings from `T-3d` or earlier;
- the 1-week model can use readings from `T-1w` or earlier.

Calendar features for `T` are known in advance. Target-time `PRICE_AUD_MWH` and `DEMAND_MW` are excluded because Sam would not know their realised future values. Customer ID is only used for grouping and building-balanced evaluation.

The teammate code builds lags using row shifts. This program rebuilds all lags using exact customer/timestamp lookup, so a missing 30-minute interval cannot silently become a false lag. Differences are saved in `lag_audit_*.csv`.

## Free guesses

Each horizon has its matching no-cost baseline:

- 1 hour: use consumption from 1 hour before the target;
- 3 days: use consumption from 3 days before the target;
- 1 week: use consumption from 1 week before the target.

The headline metric is macro MAE: calculate MAE separately for every building, then average the building results. `results.csv` states whether each ML model beats the matching free guess.

## Outputs

- `results.csv`: the only source for headline report numbers and Table 2
- `predictions.csv`: long-form building, time, distance, model, actual use and forecast
- `predictions_wide.csv`: one row per building/time/distance with all three forecasts, convenient for charts
- `per_building_metrics.csv`: building-balanced error analysis
- `validation_trials.csv`: every setting tried and the selected setting
- `selected_settings.json`: settings locked before test access
- `cohort.csv`: exact reproducible customer IDs used in each split
- `manual_lag_check_finetune.csv`: ten timestamp-level clues rebuilt for one building
- `feature_importance.csv`: inputs for Joseph's Figure 7
- `lag_audit_final.csv`: check against teammate-provided lag columns
- `models/*.joblib`: six fitted models

Create Table 2 strictly from the results file:

```bash
python generate_table2.py
python generate_headline_results.py
python verify_lag_spot_check.py
python generate_run_report.py
python run_analysis_starter.py
python verify_saved_results.py
```

Open `notebooks/predictions_starter.ipynb` for the starter analysis handed to Joseph and Regina.

After the locked final run, `generate_run_report.py` creates `report_sections.md`, an executed-settings appendix and `HANDOVER.md`. It joins the saved household metrics with published dwelling types to create `per_building_metrics_with_type.csv` and `error_by_dwelling_type.csv`. The starter notebook reads these saved exports and creates starter figures without retraining. `verify_lag_spot_check.py` independently checks one fine-tuning household's clues against the raw source file.

The command `python run_analysis_starter.py` executes the bundled notebook cells and saves the starter figures without requiring Jupyter. To use the notebook interactively, install `notebook` and `ipykernel` separately or open it in a configured VS Code notebook environment.

`python verify_saved_results.py` independently checks the saved per-household metrics, aggregate results and every long-form forecast against the wide export. It writes `verification_summary.json` without opening source test data or fitted models. XGBoost wide forecasts are read at their original float32 precision to reproduce pre-export metrics; the canonical long export contains the full promoted values.

## Test

```bash
pytest -q
```

The tests include hand-calculated MAE/RMSE/NMAE, a missing-interval leakage check, horizon-specific feature availability, both gradient-boosting libraries, all three distances, prediction export, and the final-test lock.
