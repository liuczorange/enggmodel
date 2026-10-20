# Handover notes and proposal changes

## What is being predicted

The target is interval electricity consumption (`KWH_READING`, kWh). It is not an electricity bill or electricity price. `PRICE_AUD_MWH` is the regional wholesale price and is a separate variable.

## Differences identified in the teammate data repository

1. **Dataset scope changed.** The teammate pipeline uses Smart Grid Smart City residential customer load profiles plus AEMO market data. This is different from the earlier BDG2 non-residential-building project. Report language must consistently say residential customers/households unless the team has a defensible mapping from customers to buildings.

2. **Building type is not present.** The exported schema has `CUSTOMER_ID` but no dwelling/building type. Joseph's planned “error by building type” figure cannot be supported from the current three modelling files. The team must either join an appropriate field from the SGSC index/demographic files before the final run, or replace that figure with a pre-agreed customer/cluster analysis. Do not infer types from customer IDs.

3. **The split is not reproducible if regenerated.** `src/unify.py` calls `np.random.shuffle` without a fixed seed. The current output files can be frozen, but rerunning the preprocessing may assign different customers to learn, fine-tune and test. Recommended upstream fix: initialise `np.random.default_rng(42)` and shuffle a copied, sorted customer array. The modelling pipeline stores SHA-256 hashes and refuses final evaluation if learn/fine-tune files change after tuning.

4. **Row-shift lags can be wrong across gaps.** Upstream lags use `groupby().shift()`, which assumes every half-hour row is present. The modelling program rebuilds lags by exact customer and timestamp and records agreement with upstream columns in `lag_audit_*.csv`.

5. **Future market values are unavailable.** The row's `PRICE_AUD_MWH` and `DEMAND_MW` occur at the target time. Unless separate price/demand forecasts are supplied, using realised target-time values would leak future information. They are excluded from the initial model.

6. **The AEMO download and cleaning paths do not currently match.** In the teammate repository, `fetch_aemo.py` writes `data/aemo_nsw_2013.csv`, while `clean_aemo.py` expects `data/raw/aemo_nsw_2013.csv`. Before regenerating the cleaned data, make both scripts use the same path (preferably `data/raw/aemo_nsw_2013.csv`) or move the downloaded file there. This does not affect already-frozen cleaned CSV files.

## Local preparation from public sources

`download_energy_data.py` and `prepare_energy_data.py` reproduce the teammate's source/schema without requiring their private local outputs. They retain sampling seed 123, 1,500 customers and the 70/15/15 split proportions, fix the split seed at 42, and compute exact timestamp lags. The split differs from an unknown previous unseeded teammate run; the team should adopt one frozen split before combining results. Processing is per customer to reduce memory use. Source checksums and output IDs/hashes are recorded in `data/cleaned/preparation_manifest.json`.

The downloaded demographic index resolves item 2 above: `customer_metadata.csv` exports the publisher's house/unit cluster assignments plus asserted/reported dwelling fields, with unknown classifications retained. Joseph can join on `building_id`. These fields are for analysis only. Full-year energy/peak summaries and cluster labels are excluded from model inputs.

The published SGSC file already contains filled missing readings. Because the source imputation may use future observations, the report must describe this as retrospective evaluation of a processed dataset and retain this as a leakage limitation. SGSC and AEMO interval-label/DST alignment remains an upstream assumption; future realised market values are excluded from forecasting.

## Required handover after the locked final run

- `results.csv` and generated `table2.md` to the report lead.
- `predictions.csv` and `notebooks/predictions_starter.ipynb` to Joseph and Regina.
- `validation_trials.csv`, `selected_settings.json` and this change note to Regina.
- `lag_audit_final.csv` to the data-processing owner for reconciliation.
