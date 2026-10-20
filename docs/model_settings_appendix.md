# Appendix: model settings and execution

## Fixed experiment settings

- Random seed: 42
- Reproducible cohort caps: 100 learn, 30 fine-tune, 30 test customers
- Observation frequency: 30 minutes
- Horizons: 1 hour, 3 days, 1 week
- Selection metric: macro building-level MAE on the fine-tuning group
- Training groups: `data_learn.csv`
- Fine-tuning groups: `data_finetune.csv`
- Final test groups: `data_test.csv`
- Final model fit: learn + fine-tune groups, after settings are selected
- Negative predictions: clipped to zero before evaluation
- Parallelism: all available CPU threads (`n_jobs=-1`)

## LightGBM settings tried

All trials use `objective="regression_l1"`.

| Trial | learning_rate | num_leaves | max_depth | n_estimators | min_child_samples |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 15 | 6 | 250 | 30 |
| 2 | 0.04 | 31 | 8 | 350 | 40 |
| 3 | 0.08 | 15 | 5 | 200 | 50 |

## XGBoost settings tried

All trials use `objective="reg:absoluteerror"`.

| Trial | learning_rate | max_depth | n_estimators | min_child_weight |
|---:|---:|---:|---:|---:|
| 1 | 0.05 | 5 | 250 | 3 |
| 2 | 0.04 | 7 | 350 | 5 |
| 3 | 0.08 | 4 | 200 | 5 |

The authoritative trial record is generated as `validation_trials.csv`; this appendix must be checked against that file before submission.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download official sources and create the teammate-compatible CSV schema.
python download_energy_data.py
python prepare_energy_data.py

python run_multihorizon.py --stage tune --horizon all \
  --data-dir /path/to/energy-building-data-processing/data/cleaned

# Run exactly once, after checking validation_trials.csv and selected_settings.json.
python run_multihorizon.py --stage final --horizon all \
  --data-dir /path/to/energy-building-data-processing/data/cleaned

python generate_table2.py
python generate_headline_results.py
python verify_lag_spot_check.py
python generate_run_report.py
python run_analysis_starter.py
```

`run_analysis_starter.py` executes the bundled notebook cells and saves starter figures using the listed project dependencies. For an interactive notebook UI, separately install `notebook` and `ipykernel`, or use a configured Jupyter environment in VS Code.

## Output provenance

The report must source all headline values from `multihorizon_outputs/results.csv`. Table 2 must be regenerated using `generate_table2.py`; values must not be copied from console output. Prediction-based figures must read `predictions.csv`.
