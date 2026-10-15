# Jump-Aware Adaptive Learning for Non-Stationary Demand Forecasting

CPU-first, one-step-ahead daily sales forecasting for each `(store_nbr, family)` series in the Kaggle Corporación Favorita dataset. Each date is forecast as a batch; actual sales for that date are revealed only after the whole batch has been predicted and can update adapters for the next date.

## Current implementation status

The repository contains data collection, causal preprocessing, chronological splitting, LightGBM training, residual filtering, synthetic experiments, EDA, and tests. Completed experiments cover all 1,782 Favorita series, all 12 Beijing stations, and a full USGS feature ablation. The USGS validation gain did not generalize to test; measured results and limitations are recorded below. Raw data, models and large predictions stay local in Git-ignored directories.

## Data and setup

Use Python 3.11+. Put Kaggle `train.csv` (required), and optionally `stores.csv`, `holidays_events.csv`, and `oil.csv`, in `data/raw/`. This workspace's `train.csv` was obtained from a public [Hugging Face mirror](https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting) because Kaggle credentials were not configured; the raw-data directory is Git-ignored. The scripts do not download files automatically. `pyarrow` is optional and enables Parquet output. Without it, preparation saves compressed CSV.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The default split is train through `2015-12-31`, validation `2016-01-01`–`2016-03-31`, and test from `2016-04-01` through the last labeled date. The loader checks the actual date coverage and rejects empty splits. It does not use Kaggle's unlabeled `test.csv`. `requirements-lock.txt` records the tested Python 3.11 environment; use `pip install -r requirements-lock.txt` for the same package versions. Tests work without downloaded datasets; the real Beijing coverage integration check is skipped when that dataset is absent.

## Run

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.run_eda
.\.venv\Scripts\python.exe -m scripts.run_experiments --seed 42
.\.venv\Scripts\python.exe -m scripts.run_baselines --max-series 100 --seed 42
```

Omit `--max-series` for all available series. The subset is chosen deterministically from train series and carried across all splits. `configs/default.json` controls dates, seed, thread count, and the fixed LightGBM parameters. Generated artifacts go under `results/` and are ignored by Git.

`run_baselines.py` fits one offline LightGBM model on train rows, uses validation for early stopping and global filter parameter selection, freezes those settings, then evaluates all baselines on the same test rows. Validation grid search is limited to the first 100 sorted series for runtime; it does not inspect test labels. Test-time lag features use actual labels from earlier dates according to the one-step online feedback assumption. For each date, all residual-adapter predictions are formed before any same-date labels update state.

## Method

The target is `z[g,t] = log(1 + sales[g,t])`. The fixed predictor is `f_phi(x[g,t])`; residual correction estimates a latent process:

```text
z[g,t] = f_phi(x[g,t]) + theta[g,t] + epsilon[g,t]
theta[g,t] = mu + rho (theta[g,t-1] - mu) + eta_t + I_t J_t
eta_t ~ N(0,q), I_t ~ Bernoulli(p), J_t ~ N(0,q_jump), epsilon_t ~ N(0,R)
```

This is a discrete OU model with a compound-Poisson jump approximation. `JumpAwareFilter` computes no-jump and jump Gaussian branches and moment-matches them; it is an approximate two-component filter, not exact full Bayesian filtering. Mixture weights use log likelihoods. Each series has its own posterior state while parameters are global. The filter stores the prior/corrected estimate and estimated jump probability through its update return value.

Features include calendar values, promotion, optional store/holiday metadata, prior-day oil price, and per-series exact calendar-day lags/time-window rolling statistics that exclude the current date. `transactions` and same-day oil are excluded. Missing records are retained as missing and are not silently imputed as zero. Predictions are clipped at zero after inverse log transform.

The synthetic harness covers stationary, Gaussian-drift, jump-only, mixed, and isolated-outlier settings. Jump detection metrics only make sense there because the simulated latent state and jumps are known. For real data, residual spikes are not treated as ground-truth regime shifts.

## Metrics and artifacts

Test metrics include RMSLE, MAE, log-RMSE, and mean per-series MAE. Predictions are written with `date`, `store_nbr`, `family`, `actual`, `prediction`, and `model`, plus adapter diagnostics where available. EDA writes summary tables and plots to `results/eda/`; synthetic runs write per-step traces and `synthetic_metrics.csv` to `results/synthetic/`; baseline outputs include metrics, predictions, rolling-error, forecast-versus-actual and jump-diagnostic plots under `results/baselines/`.

## Diagnostics

Run `\.venv\Scripts\python.exe -m scripts.run_diagnostics --max-series 100 --seed 42` to audit the saved smoke run, create validation/test residual diagnostics, repeat five-seed synthetic checks, and evaluate the fixed-parameter `p`/`q_jump` grid on validation only. Outputs are saved under `results/diagnostics/`; start with `diagnostics.md`. The command reuses a saved LightGBM model when its manifest matches, otherwise reproduces the model from the archived config and verifies it against the saved test predictions.

The baseline script provides an initial pipeline and matched test predictions. Deeper validation still needed on a real dataset includes full-budget runtime, validation choice stability across series/seeds, feature coverage and holiday semantics, and review of the produced diagnostics. Do not interpret the synthetic experiment or a small `--max-series` smoke run as evidence of real-data superiority.

## Model review

Run `\.venv\Scripts\python.exe -m scripts.run_model_review --max-series 100 --seed 42` to compare the original two-branch jump filter with a three-branch robust filter that separates persistent latent jumps from one-day observation outliers. It reuses Task 02's saved LightGBM predictions, tunes bounded filter grids on validation only, uses disjoint synthetic calibration/evaluation seeds, and evaluates the selected filters once on the previously inspected test split. Review `results/model_review/review.md`; grid, selected configs, synthetic metrics/plots and test predictions are saved alongside it. No LightGBM retraining or extra package installation is required.

## First real-data smoke run

On 2026-10-13, the pipeline ran on 100 deterministically selected series, seed 42, and the default chronological dates. This is a pipeline smoke result, not a full-data comparison:

| Model | Test RMSLE | Test MAE |
|---|---:|---:|
| Fixed LightGBM | 0.41987 | 102.04 |
| LightGBM + Kalman | 0.42047 | 102.51 |
| LightGBM + Jump filter | 0.42047 | 102.51 |
| LightGBM + EMA | 0.43040 | 97.64 |
| Seasonal naive | 0.56698 | 110.79 |

Validation selected `p=0` for this subset, so the jump filter reduced to its Gaussian counterpart on test. Scores are saved in `results/baselines/metrics.csv`; predictions and diagnostic plots are alongside them. These measurements do not establish that any adaptive method improves the full Favorita benchmark.

## Independent Beijing validation (Task 04)

Download the [official UCI Beijing Multi-Site Air Quality dataset](https://archive.ics.uci.edu/dataset/501/beijing) (Chen, 2017; [DOI 10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0), extract its inner `PRSA2017_Data_20130301-20170228.zip` under `data/raw/beijing/`, preserving the 12 station CSV names. The raw files and generated artifacts are Git-ignored. No extra package is required.

The CLI runs validation selection and locked test evaluation in separate stages. Test execution refuses a changed dataset/model hash or an existing test result:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_external_validation --stage validate
.\.venv\Scripts\python.exe -m scripts.run_external_validation --stage test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The protocol forecasts PM2.5 at all 12 stations for an hour before any current-hour label updates a filter. LightGBM uses calendar features and exact-hour lagged measurements; filter configurations are selected on January–June 2016 and locked before the July 2016–February 2017 test. Start with `results/external_validation/report.md` for provenance, missingness, validation grid, test comparisons, paired seven-day block intervals, and data-mining figures. The paired test forecasts are in `test_predictions.csv.gz`. `--stage report` rebuilds only descriptive outputs from these saved forecasts. This independent dataset is separate from the previously inspected Favorita test.

## BTL report from locked artifacts (Task 05)

```powershell
.\.venv\Scripts\python.exe -m scripts.build_report
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The report builder audits saved predictions and metrics from the 100-series Favorita comparison, the full 12-station Beijing locked test, and the ten-seed synthetic evaluation. It checks source/model hashes, preserves the existing test lock, and writes only under `reports/`. Start with `reports/BTL_REPORT_DRAFT.md`; `reports/evidence_audit.md` records the provenance and timing assumptions. The report treats Favorita test scores as exploratory and does not use saved test results to select new parameters.

## Full Favorita exploratory run (all 1,782 series)

The later full-cohort run is separate from Task 01–04 outputs. It trains one fixed LightGBM on all 1,945,944 training rows, selects 39 filter candidates on 162,162 validation rows, locks the model/config/source hashes, then evaluates 892,782 test rows once. Run only in this order:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_full_favorita --stage validate
.\.venv\Scripts\python.exe -m scripts.run_full_favorita --stage test
.\.venv\Scripts\python.exe -m scripts.build_report
```

The completed first run is in `results/full_favorita_exploratory/`; the first two commands refuse to repeat after their lock/output exists. See `reports/full_favorita_addendum.md` for its validation, test, paired uncertainty, and limits. These scores remain exploratory because the Favorita test dates were inspected in prior tasks. The report builder now audits this run as well as the older artifacts, and the BTL draft links to the addendum.

## Web data collection: USGS earthquake catalog

The project also collects external data through the public [USGS FDSN API](https://earthquake.usgs.gov/fdsnws/event/1/), using only Python's standard HTTP client. This is web collection via API, rather than HTML scraping. Favorita supplies the sales labels; the independently collected source supplies earthquake events in a rectangle around mainland Ecuador and neighboring/offshore areas.

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_external_data --stage collect --snapshot data/raw/external/usgs_ecuador/2026-10-14
.\.venv\Scripts\python.exe -m scripts.collect_external_data --stage prepare --snapshot data/raw/external/usgs_ecuador/2026-10-14
.\.venv\Scripts\python.exe -m scripts.collect_external_data --stage report --snapshot data/raw/external/usgs_ecuador/2026-10-14
```

`--stage all` runs those stages in order. Collection settings are in `configs/collection_usgs.json`. Raw yearly GeoJSON responses and metadata preserve URLs, retrieval timestamps, status codes, and SHA-256 hashes; cached files are verified and reused. `prepare` and `report` work offline. Use a new snapshot directory to deliberately collect a later catalog revision.

The first snapshot contains 184 events (magnitude >= 4.5), spanning the 2013-01-01–2017-08-15 study period plus seven lookback days, and produces 1,688 calendar days. The descriptive daily join covers all 3,000,888 sales rows through 1,684 observed sales dates; four absent sales dates remain missing. Normalized tables are under `data/processed/external/usgs_ecuador/2026-10-14/`; descriptive outputs are under `results/data_collection/usgs_ecuador/2026-10-14/`. Start with `reports/DATA_COLLECTION.md` for data quality, train-only associations, and the acquisition/preprocessing/feature pipeline.

The catalog is retrospective and revised: origin timestamps and last-update timestamps do not establish historical first-publication times. Prior-day earthquake features are evaluated in the separate full USGS ablation below, with availability explicitly unverified. They were not added to the earlier Task 01–05 comparisons. Country-level catalog events are not physical jump labels, store-level ground shaking, or causal sales effects.

## Full model ablation with collected USGS data

```powershell
.\.venv\Scripts\python.exe -m scripts.run_usgs_ablation --stage validate
.\.venv\Scripts\python.exe -m scripts.run_usgs_ablation --stage test
.\.venv\Scripts\python.exe -m scripts.run_usgs_ablation --stage report
```

`configs/usgs_ablation.json` specifies a full-cohort 2×2 comparison: the same 12 baseline features, with/without categorical store and family IDs, and with/without the four strictly prior-day USGS features. Each model has the same seed, LightGBM budget, and chronological split. Two convex blends learn their weights from validation RMSLE before test. Training uses all 1,945,944 training rows, validation uses 162,162 rows, and test uses 892,782 rows across all 1,782 series.

Validation writes model, code, data, and prediction hashes to `results/usgs_ablation_exploratory/selected_configs.json`. Test verifies the lock and writes a start marker to prevent repeated model evaluation. `--stage report` only audits and summarizes saved predictions. The collection remains retrospective, with historical first-publication times unverified; the Favorita test period was inspected in earlier tasks. Start with `reports/USGS_MODEL_ABLATION.md` for measured results, validation selection, ensemble weights, uncertainty, and provenance. Earlier Task 01–05 model artifacts are preserved.

The completed run selected LightGBM + USGS on validation (RMSLE 0.455867 vs 0.464533 for LightGBM), but its test RMSLE was higher (0.439166 vs 0.437766). Adding IDs also did not improve RMSLE in this budget. Both convex blends selected beta=1 and matched their USGS parent exactly. This is a negative exploratory finding for the added data, not evidence of an accuracy gain. The report builder independently recomputes saved test metrics and paired intervals and links these findings into the BTL draft.

## Mathematical residual calibration (new exploratory method)

`src/hierarchical_residual.py` implements a strictly convex Gaussian
partial-pooling correction for store, family, and store-family residual bias.
An auxiliary LightGBM trained on 2013–2014 supplies out-of-sample residuals
for 2015. Ridge penalties are selected on Jan–Feb 2016; March is reported
without affecting the selection rule. The fixed original LightGBM provides
the base validation and test forecasts. The one-time test stage reads saved
base predictions and refuses repetition after writing its marker.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_hierarchical_residual --stage develop
.\.venv\Scripts\python.exe -m scripts.run_hierarchical_residual --stage test
.\.venv\Scripts\python.exe -m scripts.run_online_gate_development
.\.venv\Scripts\python.exe -m scripts.report_hierarchical_residual
```

The completed test is **exploratory** because earlier project stages already
inspected the Favorita test period. The correction reduced test RMSLE from
0.437766 to 0.437142, while MAE increased from 77.816 to 77.844. Its exact
quadratic solution, conditional Bayes argument, algebraic risk decomposition,
ablation, and limits are in [the mathematical method report](reports/MATHEMATICAL_METHOD.md).
The online OGD/ridge gates are retained as validation-only negative branches.
These methods use supervised feedback; reinforcement learning would add a
decision process that this one-step forecasting task does not contain.

## Implementation optimization and historical reproducibility

Feature construction fills positional arrays instead of repeatedly assigning
DataFrame slices per series. Online backtests forecast and update whole date
batches using the existing vectorized filters; scalar adapters remain supported.
Metric aggregation computes log error once and uses compact group codes. See
[measured timing and regression checks](reports/OPTIMIZATION.md).

The original USGS code is byte-preserved under
[`provenance/ede2f98b9ae4/`](provenance/README.md). Report-only audits may verify
that historical snapshot when current code has been optimized. Prediction/test
still enforce current-code hashes and the existing test-start marker. New
experiments require a separate output directory and a fresh validation lock.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.benchmark_optimizations
.\.venv\Scripts\python.exe -m scripts.build_report
```

The benchmark needs local Favorita data and saved USGS validation predictions;
it compares implementations without retraining models or evaluating a new test.
Reports and small figures are versioned; links to raw data/model/prediction
artifacts refer to local files that are intentionally excluded from Git.

Known modeling limitation for later discussion: Beijing's current missing-target
path skips measurement updates but does not commit every predict-only time
transition. Historical scores describe that implementation. A corrected gap-aware
filter needs a separately versioned evaluation; the performance-only optimization
does not change this behavior or revise those scores.
