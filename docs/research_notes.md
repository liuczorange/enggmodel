# Research notes for the oral and report

## Gradient boosting in one minute

Start with a deliberately simple prediction, usually a constant. Measure each training example's remaining error under the chosen loss. Fit a small decision tree to the direction that would reduce those errors, add only a fraction of that tree's correction using the learning rate, and repeat. The final forecast is the sum of many weak trees. Trees supply nonlinear thresholds and interactions; boosting makes them cooperate by concentrating each new tree on what the current ensemble still gets wrong.

Technically, gradient boosting performs stage-wise optimisation in function space: each learner approximates a negative loss gradient. Our absolute-error objectives emphasise reducing typical absolute misses rather than heavily squaring rare large misses. The foundational source is Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, *Annals of Statistics* 29(5), 2001, DOI [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).

## LightGBM versus XGBoost

Both are gradient-boosted tree libraries, so comparing them is not a comparison between unrelated model families. It is a robustness check across two mature implementations with different tree-building and systems choices.

- **XGBoost:** emphasises a regularised objective and scalable approximate tree construction, including sparsity-aware split finding, a weighted quantile sketch, cache-aware computation and distributed/out-of-core operation. Its common `depthwise` growth expands a level before going deeper, although other growth policies exist. Source: Chen and Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016, DOI [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- **LightGBM:** uses histogram-based learning and is best known for Gradient-based One-Side Sampling (retaining examples with large gradients) and Exclusive Feature Bundling (combining sparse, mutually exclusive features). Its leaf-wise growth expands the leaf with the largest estimated gain, which can be efficient and accurate but makes depth/leaf controls important. Source: Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017, [official paper page](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).

**Fair-comparison sentence for the oral:** “We gave both libraries the same leakage-safe inputs, customer splits, three forecast horizons and macro-MAE selection rule; then we tuned only a small documented grid for each. Agreement strengthens the conclusion, while disagreement tells us the result depends on implementation or regularisation choices.”

Distinguish library capabilities from executed settings: the current LightGBM estimator uses its default `gbdt` boosting mode, not an explicitly enabled GOSS sampling strategy. The original paper describes GOSS/EFB, but this experiment does not isolate or benchmark those techniques. The same number of configurations is a documented practical comparison, not proof of equal compute or equal tuning coverage.

## Why longer-horizon forecasts usually get worse

At one hour, the model has a reading close to the target and can exploit strong short-term autocorrelation. At three days, the newest admissible demand is three days old; at one week it is one week old. More occupancy, behaviour, weather and appliance events can change in the gap, so conditional uncertainty grows and the model relies increasingly on repeating calendar patterns. This project makes that comparison strict by removing any lag that occurs after the forecast origin.

Do not claim that error *must* increase monotonically in every dataset. Weekly regularity can occasionally make a one-week seasonal forecast competitive with an awkward three-day horizon. State what `results.csv` shows, then explain it.

## Free guesses used by other studies

Common no-training comparisons are:

- persistence: use the most recent observed load;
- seasonal persistence/seasonal naive: use load at the same time one day or one week earlier;
- simple historical averages, sometimes separated by hour and weekday/weekend;
- classical exponential smoothing or autoregressive models as stronger statistical baselines.

A 2023 multi-horizon building-energy study explicitly used 24-hour and 168-hour seasonal-naive baselines, which supports day/week historical guesses as credible minimum benchmarks: *A study of deep learning-based multi-horizon building energy forecasting*, *Energy and Buildings* (2023), DOI [10.1016/j.enbuild.2023.113810](https://doi.org/10.1016/j.enbuild.2023.113810).

Our matching free guesses are intentionally operational: target minus 1 hour, 3 days or 1 week. They answer, “Does the paid modelling effort improve on simply reusing the last value available at that same forecast distance?”

## Statements to avoid

- Do not say LightGBM is always more accurate or always faster than XGBoost; these are empirical outcomes that depend on data and settings.
- Do not say feature importance proves causation.
- Do not describe pooled R² as building-balanced; macro MAE/RMSE are the building-balanced measures.
- Do not report test results from console text or a notebook copy. Use `results.csv` and regenerate Table 2.
- Do not call `PRICE_AUD_MWH` the target or describe kWh as dollars.
