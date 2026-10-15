"""Locked, two-stage external validation on official UCI Beijing air quality."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.beijing_backtest import NAMES, online_backtest, score
from src.beijing_data import (END, MEASURES, NONNEGATIVE, TRAIN_END, VAL_END,
                              load_beijing, make_beijing_features, split_beijing)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/beijing"
OUT = ROOT / "results/external_validation"
MODEL = OUT / "beijing_lightgbm.txt"
LOCK = OUT / "selected_configs.json"
SEED = 42


def markdown_table(frame: pd.DataFrame, index: bool = False, digits: int = 3) -> str:
    """Small Markdown renderer without an optional tabulate dependency."""
    rows = frame.reset_index() if index else frame
    columns = list(rows.columns)
    def cell(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(map(str, columns)) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in rows.itertuples(index=False, name=None))
    return "\n".join(lines)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def candidates(residual_variance: float) -> dict[str, list[dict]]:
    # Bounded, shared process grid. The validation residual scale only fixes
    # units; no test data participates in constructing or selecting candidates.
    r = max(0.03, float(residual_variance))
    common = [dict(rho=rho, q=q, R=r*factor) for rho in (0.8, 0.98)
              for q in (0.002, 0.03) for factor in (0.5, 2.0)]
    out = {"LightGBM + EMA": [{"alpha": a} for a in (0.01, 0.05, 0.2, 0.5)],
           "LightGBM + Kalman": common}
    out["LightGBM + Robust Kalman"] = [dict(c, p_outlier=p, R_out_ratio=25)
                                            for c in common for p in (0.0, 0.02)]
    out["LightGBM + Jump Filter"] = [dict(c, p_jump=p, q_jump=0.5)
                                     for c in common for p in (0.0, 0.02)]
    out["LightGBM + Robust Jump Filter"] = [dict(c, p_jump=pj, q_jump=0.5,
                                                   p_outlier=po, R_out_ratio=25)
                                            for c in common for pj, po in
                                            ((0.0, 0.0), (0.02, 0.0), (0.0, 0.02), (0.02, 0.02))]
    return out


def fit_and_lock() -> None:
    if LOCK.exists():
        raise FileExistsError(f"Existing test lock at {LOCK}; use --stage test or a new output directory")
    raw = load_beijing(RAW, through=VAL_END)
    frame, columns = make_beijing_features(raw)
    splits = split_beijing_partial(frame)
    train = splits["train"]
    validation = splits["validation"].sort_values(["timestamp", "station"]).reset_index(drop=True)
    fit = train.loc[train["PM2.5"].notna()].copy()
    model = lgb.LGBMRegressor(objective="regression", n_estimators=350, learning_rate=0.04,
                              num_leaves=31, min_child_samples=40, max_depth=-1,
                              colsample_bytree=0.85, random_state=SEED,
                              n_jobs=4, verbosity=-1)
    model.fit(fit[columns], np.log1p(fit["PM2.5"]),
              categorical_feature=["station", "wd_lag1"])
    base = model.predict(validation[columns])
    if not np.isfinite(base).all():
        raise FloatingPointError("Non-finite base predictions")
    fallback = train.groupby("station", observed=True)["PM2.5"].median().to_dict()
    valid = validation["PM2.5"].notna().to_numpy()
    residual_var = np.var(np.log1p(validation.loc[valid, "PM2.5"].to_numpy()) - base[valid])
    grid, chosen, val_metrics = [], {}, []
    for family, configs in candidates(residual_var).items():
        best = None
        for i, config in enumerate(configs):
            pred = online_backtest(validation, base, family, config, fallback)["prediction"]
            met = score(validation["PM2.5"], pred)
            grid.append({"model": family, "candidate_id": i, "config": json.dumps(config, sort_keys=True), **met})
            rank = (met["rmse_log"], met["rmse"], i)
            if best is None or rank < best[0]:
                best = (rank, config, met)
        chosen[family] = best[1]
        val_metrics.append({"model": family, "candidate_count": len(configs), **best[2]})
        print(f"Selected {family}: {best[1]} log RMSE={best[2]['rmse_log']:.5f}", flush=True)
    for family in NAMES[:2]:
        pred = online_backtest(validation, base, family, station_fallback=fallback)["prediction"]
        val_metrics.append({"model": family, "candidate_count": 0, **score(validation["PM2.5"], pred)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(OUT / "validation_grid.csv", index=False)
    pd.DataFrame(val_metrics).to_csv(OUT / "validation_metrics.csv", index=False)
    model.booster_.save_model(str(MODEL))
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
            "source_zip_sha256": sha256(RAW / "PRSA2017_Data_20130301-20170228.zip"),
            "model_sha256": sha256(MODEL), "features": columns,
            "station_fallback_train_median": fallback, "selected_configs": chosen,
            "candidate_counts": {k: len(v) for k, v in candidates(residual_var).items()},
            "selection_metric": "validation rmse_log; tie: original RMSE, then candidate order",
            "state": "independent zero-mean zero-variance at each split start; no train/validation residual warmup",
            "lightgbm": {"n_estimators": 350, "learning_rate": 0.04, "num_leaves": 31,
                         "min_child_samples": 40, "colsample_bytree": 0.85, "n_jobs": 4}}
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Locked selections: {LOCK}", flush=True)


def split_beijing_partial(frame):
    t = frame.timestamp
    train = frame.loc[t < TRAIN_END].copy()
    validation = frame.loc[(t >= TRAIN_END) & (t < VAL_END)].copy()
    if train.empty or validation.empty or train.timestamp.max() >= validation.timestamp.min():
        raise ValueError("Invalid train/validation split")
    return {"train": train, "validation": validation}


def paired_bootstrap(wide: pd.DataFrame, reps=500) -> pd.DataFrame:
    actual = wide.actual.to_numpy(float)
    good = np.isfinite(actual) & (actual >= 0)
    block = ((wide.timestamp - wide.timestamp.min()).dt.days // 7).to_numpy()
    blocks = np.unique(block)
    n = np.bincount(block[good], minlength=len(blocks))
    squared = {}
    for family in NAMES:
        p = wide[family].to_numpy(float)
        squared[family] = np.bincount(block[good], weights=(p[good]-actual[good])**2,
                                      minlength=len(blocks))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(blocks), size=(reps, len(blocks)))
    den = n[draws].sum(axis=1)
    rmses = {k: np.sqrt(v[draws].sum(axis=1)/den) for k, v in squared.items()}
    rows = []
    for family in NAMES:
        if family == "LightGBM":
            continue
        references = ["LightGBM"]
        if family == "LightGBM + Robust Kalman":
            references.append("LightGBM + Kalman")
        if family == "LightGBM + Jump Filter":
            references.append("LightGBM + Kalman")
        if family == "LightGBM + Robust Jump Filter":
            references.append("LightGBM + Robust Kalman")
        for reference in references:
            point = np.sqrt(squared[family].sum()/n.sum()) - np.sqrt(squared[reference].sum()/n.sum())
            delta = rmses[family] - rmses[reference]
            lo, hi = np.quantile(delta, [0.025, 0.975])
            rows.append({"model": family, "reference": reference, "delta_rmse": point,
                         "ci_low": lo, "ci_high": hi, "blocks": len(blocks),
                         "block_hours": 168, "bootstrap_reps": reps, "seed": SEED})
    return pd.DataFrame(rows)


def profile(raw: pd.DataFrame, predictions: pd.DataFrame, wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["period"] = np.select([raw.timestamp < TRAIN_END, raw.timestamp < VAL_END],
                               ["train", "validation"], default="test")
    records = []
    for (period, station), part in raw.groupby(["period", "station"]):
        for var in (*MEASURES, "wd"):
            numeric = var != "wd"
            records.append({"period": period, "station": station, "variable": var,
                            "rows": len(part), "missing": int(part[var].isna().sum()),
                            "missing_rate": float(part[var].isna().mean()),
                            "negative": int((part[var] < 0).sum()) if numeric and var in NONNEGATIVE else 0})
    miss = pd.DataFrame(records)
    monthly = []
    raw["year_month"] = raw.timestamp.dt.to_period("M").astype(str)
    for (month, station), part in raw.groupby(["year_month", "station"]):
        for var in (*MEASURES, "wd"):
            monthly.append({"year_month": month, "station": station, "variable": var,
                            "rows": len(part), "missing": int(part[var].isna().sum()),
                            "missing_rate": float(part[var].isna().mean())})
    pd.DataFrame(monthly).to_csv(OUT / "missingness_by_month.csv", index=False)
    summary = []
    for period, part in raw.groupby("period"):
        y = part["PM2.5"].where(part["PM2.5"] >= 0)
        summary.append({"period": period, "rows": len(part), "stations": part.station.nunique(),
                        "start": part.timestamp.min(), "end": part.timestamp.max(),
                        "pm_missing_or_invalid": int(y.isna().sum()), "pm_mean": y.mean(),
                        "pm_std": y.std(), "pm_p95": y.quantile(.95)})
    data = pd.DataFrame(summary)
    train = raw.loc[raw.timestamp < TRAIN_END].copy()
    train["PM2.5"] = train["PM2.5"].where(train["PM2.5"] >= 0)
    station_rows = []
    for station, part in train.groupby("station"):
        y = part["PM2.5"]
        station_rows.append({"station": station, "train_mean": y.mean(), "train_std": y.std(),
                             "train_missing_rate": y.isna().mean(),
                             "train_lag1_corr": y.corr(y.shift(1)),
                             "train_lag24_corr": y.corr(y.shift(24))})
    groups = pd.DataFrame(station_rows)
    groups["group"] = pd.qcut(groups.train_mean.rank(method="first"), 3,
                               labels=["low", "middle", "high"]).astype(str)
    groups.to_csv(OUT / "station_train_groups.csv", index=False)
    data.to_csv(OUT / "dataset_profile.csv", index=False)
    miss.to_csv(OUT / "missingness_by_station.csv", index=False)
    return data, miss, groups


def figures(raw, wide, metrics, groups, validation_wide):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    plot = raw.copy()
    plot["PM2.5"] = plot["PM2.5"].where(plot["PM2.5"] >= 0)
    plot["season"] = plot.timestamp.dt.month.map({12:"winter",1:"winter",2:"winter",
                  3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",
                  9:"autumn",10:"autumn",11:"autumn"})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot.groupby("station")["PM2.5"].mean().sort_values().plot.barh(ax=ax[0], title="PM2.5 by station")
    plot.groupby("season")["PM2.5"].mean().reindex(["spring","summer","autumn","winter"]).plot.bar(ax=ax[1], title="PM2.5 by season")
    fig.tight_layout(); fig.savefig(folder / "station_season.png"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot.groupby(plot.timestamp.dt.hour)["PM2.5"].mean().plot(ax=ax[0], title="Hourly mean PM2.5")
    plot.groupby(plot.timestamp.dt.dayofweek)["PM2.5"].mean().plot(ax=ax[1], title="Day-of-week mean PM2.5")
    fig.tight_layout(); fig.savefig(folder / "seasonality.png"); plt.close(fig)
    base = pd.concat([validation_wide, wide], ignore_index=True)
    base["residual_log"] = np.log1p(base.actual) - np.log1p(base["LightGBM"])
    daily = base.groupby(base.timestamp.dt.floor("D"), sort=True).agg(
        pm_mean=("actual", "mean"), residual_mean=("residual_log", "mean"),
        residual_std=("residual_log", "std"))
    daily["period"] = np.where(daily.index < VAL_END, "validation", "test")
    daily["pm_roll7"] = daily.groupby("period").pm_mean.transform(lambda x: x.rolling(7, min_periods=1).mean())
    daily["residual_roll7"] = daily.groupby("period").residual_mean.transform(lambda x: x.rolling(7, min_periods=1).mean())
    daily["residual_std7"] = daily.groupby("period").residual_mean.transform(lambda x: x.rolling(7, min_periods=2).std())
    daily.to_csv(OUT / "rolling_target_residual.csv")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    daily[["pm_mean","pm_roll7"]].plot(ax=ax[0], title="Target: daily and trailing 7-day mean")
    daily[["residual_mean","residual_roll7","residual_std7"]].plot(ax=ax[1], title="LightGBM log residual: daily and trailing 7-day summaries")
    for axis in ax:
        axis.axvline(VAL_END, color="black", linestyle="--", linewidth=1)
    fig.tight_layout(); fig.savefig(folder / "rolling_target_residual.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics.set_index("model").rmse.sort_values().plot.barh(ax=ax, title="Test PM2.5 RMSE")
    fig.tight_layout(); fig.savefig(folder / "test_model_comparison.png"); plt.close(fig)
    # Test panel is descriptive after lock; station identity fixed by train mean.
    station = groups.sort_values("train_mean").iloc[len(groups)//2].station
    example = wide.loc[(wide.station.astype(str) == station) &
                       (base.timestamp < VAL_END + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(12, 4))
    for col in ("actual", "LightGBM", "LightGBM + Kalman", "LightGBM + Robust Jump Filter"):
        ax.plot(example.timestamp, example[col], label=col, linewidth=1)
    ax.set_title(f"First test week: {station}"); ax.legend(); fig.tight_layout()
    fig.savefig(folder / "forecast_example.png"); plt.close(fig)


def evaluate_locked() -> None:
    if not LOCK.exists():
        raise FileNotFoundError("Run --stage validate first")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if sha256(MODEL) != lock["model_sha256"] or sha256(RAW / "PRSA2017_Data_20130301-20170228.zip") != lock["source_zip_sha256"]:
        raise ValueError("Model/source differs from locked validation")
    if (OUT / "test_metrics.csv").exists():
        raise FileExistsError("Test already evaluated; preserving the first locked result")
    raw = load_beijing(RAW)
    frame, columns = make_beijing_features(raw)
    if columns != lock["features"]:
        raise ValueError("Feature list changed after lock")
    test = split_beijing(frame)["test"].sort_values(["timestamp", "station"]).reset_index(drop=True)
    booster = lgb.Booster(model_file=str(MODEL))
    base = booster.predict(test[columns], num_threads=4)
    wide = test[["timestamp", "station", "PM2.5"]].rename(columns={"PM2.5":"actual"}).copy()
    wide["station"] = wide.station.astype(str)
    records = []
    for family in NAMES:
        result = online_backtest(test, base, family, lock["selected_configs"].get(family),
                                 lock["station_fallback_train_median"], diagnostics=True)
        wide[family] = result["prediction"]
        record = score(wide.actual, result["prediction"], wide.station)
        records.append({"model": family, **record})
        print(f"Test {family}: RMSE={record['rmse']:.4f}, log RMSE={record['rmse_log']:.5f}", flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    station_records = []
    for station, part in wide.groupby("station"):
        for family in NAMES:
            station_records.append({"station": station, "model": family,
                                    **score(part.actual, part[family])})
    stations = pd.DataFrame(station_records)
    stations.to_csv(OUT / "station_metrics.csv", index=False)
    predictions = wide.melt(id_vars=["timestamp", "station", "actual"], var_name="model", value_name="prediction")
    predictions.to_csv(OUT / "test_predictions.csv.gz", index=False, compression="gzip")
    postprocess_existing()


def postprocess_existing() -> None:
    """Rebuild descriptive outputs from the single saved test forecast, without re-testing."""
    if not LOCK.exists() or not (OUT / "test_predictions.csv.gz").exists():
        raise FileNotFoundError("Locked test predictions are required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (sha256(MODEL) != lock["model_sha256"] or
            sha256(RAW / "PRSA2017_Data_20130301-20170228.zip") != lock["source_zip_sha256"]):
        raise ValueError("Model/source differs from the lock")
    pred = pd.read_csv(OUT / "test_predictions.csv.gz", parse_dates=["timestamp"])
    first = pred.loc[pred.model == "LightGBM", ["timestamp", "station", "actual"]].reset_index(drop=True)
    wide = first.copy()
    for family in NAMES:
        part = pred.loc[pred.model == family].reset_index(drop=True)
        if len(part) != len(wide) or not part[["timestamp", "station"]].equals(wide[["timestamp", "station"]]):
            raise ValueError("Saved model predictions are not paired by station-hour")
        wide[family] = part.prediction.to_numpy()
    raw = load_beijing(RAW)
    data, missing, groups = profile(raw, pred, wide)
    metrics = pd.read_csv(OUT / "test_metrics.csv")
    stations = pd.read_csv(OUT / "station_metrics.csv")
    bootstrap = paired_bootstrap(wide)
    bootstrap.to_csv(OUT / "paired_bootstrap.csv", index=False)
    validation_raw = raw.loc[raw.timestamp < VAL_END]
    validation_frame, columns = make_beijing_features(validation_raw)
    validation = validation_frame.loc[validation_frame.timestamp >= TRAIN_END].copy()
    booster = lgb.Booster(model_file=str(MODEL))
    val_base = np.maximum(0, np.expm1(booster.predict(validation[columns], num_threads=4)))
    validation_wide = validation[["timestamp", "station", "PM2.5"]].rename(columns={"PM2.5":"actual"}).copy()
    validation_wide["LightGBM"] = val_base
    figures(raw, wide, metrics, groups, validation_wide)
    write_report(lock, data, missing, groups, metrics, stations, bootstrap, raw, wide)


def write_report(lock, data, missing, groups, metrics, stations, bootstrap, raw, wide):
    val = pd.read_csv(OUT / "validation_metrics.csv")
    test_table = metrics[["model", "n", "rmse", "mae", "rmse_log", "mean_station_rmse", "mean_station_mae", "mean_station_rmse_log"]]
    group_table = stations.merge(groups[["station", "group"]], on="station")
    group_table = group_table.groupby(["group", "model"], as_index=False)["rmse"].mean()
    group_table = group_table.pivot(index="group", columns="model", values="rmse").round(3)
    group_delta = group_table.drop(columns=["LightGBM"]).sub(group_table["LightGBM"], axis=0)
    group_delta.reset_index().to_csv(OUT / "group_rmse_delta.csv", index=False)
    train = raw.loc[raw.timestamp < TRAIN_END]
    pm = train.groupby("station")["PM2.5"]
    volatility = pm.apply(lambda x: x.diff().std()).round(2)
    test_missing = int(wide.actual.isna().sum())
    favor = pd.read_csv(ROOT / "results/model_review/real_test_comparison.csv")
    synthetic = pd.read_csv(ROOT / "results/model_review/synthetic_comparison.csv")
    favor_eda = pd.read_csv(ROOT / "results/eda/summary.csv").iloc[0]
    synthetic_rows = synthetic.loc[
        (synthetic.scenario.isin(["jump_only", "mixed"]) &
         (synthetic.model_family == "Robust Jump Filter") & (synthetic.parameter_mode == "oracle")) |
        ((synthetic.scenario == "isolated_outliers") &
         (synthetic.model_family.isin(["Jump Filter", "Robust Jump Filter"])) &
         (synthetic.parameter_mode == "reference")),
        ["scenario", "model_family", "parameter_mode", "prediction_rmse_log_at_0_5_mean",
         "latent_tracking_rmse_at_0_5_mean", "jump_recall_at_decision_mean",
         "false_positive_rate_at_decision_mean", "outlier_false_jump_rate_at_decision_mean"]]
    selected_jump = {k: v.get("p_jump", 0) for k, v in lock["selected_configs"].items() if "Jump" in k}
    jump_active = any(v > 0 for v in selected_jump.values())
    jump = metrics.set_index("model").loc["LightGBM + Robust Jump Filter", "rmse"]
    gaussian = metrics.set_index("model").loc["LightGBM + Robust Kalman", "rmse"]
    static = metrics.set_index("model").loc["LightGBM", "rmse"]
    contrast = bootstrap.loc[(bootstrap.model == "LightGBM + Robust Jump Filter") &
                             (bootstrap.reference == "LightGBM + Robust Kalman")].iloc[0]
    outlier_contrast = bootstrap.loc[(bootstrap.model == "LightGBM + Robust Kalman") &
                                     (bootstrap.reference == "LightGBM + Kalman")].iloc[0]
    missing_summary = missing.groupby("variable", as_index=False)[["rows", "missing", "negative"]].sum()
    missing_summary["missing_rate"] = missing_summary["missing"] / missing_summary["rows"]
    if jump_active and jump < gaussian and jump < static and contrast.ci_high < 0:
        verdict = "SUPPORTED ON EXTERNAL DATA"
    else:
        verdict = "CONDITIONAL SYNTHETIC BENEFIT"
    text = ["# Task 04 external validation — Beijing Multi-Site Air Quality", "",
        f"**Conclusion: {verdict}**", "",
        "## Provenance and protocol", "",
        "Source: [Chen (2017), UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing), DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0. The official inner ZIP SHA-256 is `" + lock["source_zip_sha256"] + "`.", "",
        f"Lock created at `{lock['locked_at_utc']}` before test evaluation. Model SHA-256 `{lock['model_sha256']}`. Seed {SEED}; CPU LightGBM, 4 threads. LightGBM fit once on nonmissing train targets (fixed 350 trees); no test fit, search, or warmup.", "",
        "Train: 2013-03-01 through 2015-12-31; validation: 2016-01-01 through 2016-06-30; test: 2016-07-01 through 2017-02-28. The source timestamps are local hourly fields with no UTC conversion. All station forecasts at an hour precede target updates. Validation and test filters each start at zero mean and zero variance; missing targets cause no update. The seasonal naive uses exact lag 24h, then lag 1h, then a train-only station median.", "",
        f"Features ({len(lock['features'])}): `" + "`, `".join(lock["features"]) + "`. All measured pollutant/weather/wind variables are lagged at least one exact calendar hour. PM rolling 24h/168h mean and std use only shifted history. Missing features stay NA with indicators; LightGBM handles NA. Nonnegative physical variables with negative values are marked missing; negative temperatures/dew points remain valid. No current-hour observation from any station enters a feature.", "",
        "RMSLE equals log RMSE here: `sqrt(mean((log1p(max(pred,0))-log1p(actual))^2))`. RMSE/MAE are in original PM2.5 µg/m³. Micro metrics pool all valid station-hours; mean-station metrics average 12 separate station scores. Candidate selection minimizes validation log RMSE with original RMSE and candidate order as ties.", "",
        "## Coverage and data mining", "", markdown_table(data), "",
        f"Test target missing/invalid: {test_missing} of {len(wide)} station-hours, excluded from every metric and update. Missingness by station/variable/split is in `missingness_by_station.csv`; monthly station/variable missingness is in `missingness_by_month.csv`. Negative invalid counts by variable are included in the split table. Training-only station groups are in `station_train_groups.csv`.", "",
        "Full-period missing and physically invalid negative counts:", "", markdown_table(missing_summary), "",
        "Train station volatility (SD of consecutive hourly PM2.5 differences): " + ", ".join(f"{k}={v}" for k,v in volatility.items()) + ".", "",
        "Train-only station groups (tertiles of train PM2.5 mean):", "", markdown_table(groups.sort_values("train_mean")), "",
        "Descriptive plots: `figures/station_season.png`, `figures/seasonality.png`, `figures/rolling_target_residual.png`, `figures/test_model_comparison.png`, `figures/forecast_example.png`. `rolling_target_residual.csv` reports validation and test daily target/residual mean and trailing 7-day means/std, reset at the split boundary; test rows were produced only after lock. Spikes in residual are not ground-truth jumps.", "",
        "## Validation selection", "", markdown_table(val, digits=5), "",
        "Candidate counts: `" + json.dumps(lock["candidate_counts"], sort_keys=True) + "`. Shared bounded rho/q/R process grid and inactive/active mechanism branches are in `validation_grid.csv`. Selected parameters are frozen in `selected_configs.json`.", "",
        "Selected jump probabilities: `" + json.dumps(selected_jump, sort_keys=True) + "`.", "",
        "## Locked Beijing test", "", markdown_table(test_table, digits=5), "",
        "Paired 7-day time-block bootstrap uses all 12 stations in each block, 500 draws, seed 42. Delta RMSE is model minus reference; intervals describe sampling uncertainty, not causality:", "", markdown_table(bootstrap, digits=4), "",
        "Mean test RMSE by train-defined station group:", "", markdown_table(group_table, index=True), "",
        "Group RMSE changes relative to LightGBM (negative favors the filter):", "", markdown_table(group_delta, index=True), "",
        "## Cross-dataset interpretation", "",
        "Favorita Task 03 (100 series; previously inspected exploratory test, RMSLE/MAE):", "",
        markdown_table(favor[["model", "n", "rmsle", "mae"]], digits=5), "",
        f"Favorita EDA covered {int(favor_eda['rows']):,} rows, {int(favor_eda['stores'])} stores, {int(favor_eda['families'])} families and {int(favor_eda['series'])} series (`results/eda/summary.csv`). These scales do not compare directly with Beijing RMSE. The Task 03 validation-selected two-branch jump prior was 0.001; robust jump selected p_jump=0, p_outlier=0.005.", "",
        "Task 03 synthetic, extracted from `results/model_review/synthetic_comparison.csv` (10 evaluation seeds):", "",
        markdown_table(synthetic_rows, digits=4), "",
        "Oracle process parameters for injected jumps and reference parameters for isolated outliers differ from validation-selected settings. The old jump filter flagged all isolated injected outliers as jumps; the robust version reduced that rate. Real-world jumps have no ground-truth labels, so these detection figures do not transfer directly.", "",
        f"For Beijing, robust jump RMSE {jump:.4f}, robust Gaussian {gaussian:.4f}, static LightGBM {static:.4f}; robust-jump minus robust-Gaussian delta {contrast.delta_rmse:+.4f} (95% CI {contrast.ci_low:+.4f} to {contrast.ci_high:+.4f}). Robust Kalman minus ordinary Kalman is {outlier_contrast.delta_rmse:+.4f} (95% CI {outlier_contrast.ci_low:+.4f} to {outlier_contrast.ci_high:+.4f}), so the outlier branch has no clear independent test gain at this scale. The Gaussian correction itself improves over static LightGBM. A PM2.5 residual spike cannot verify a physical regime change or latent Lévy jump. The one-step feedback protocol evaluates short-term correction, not long-horizon forecast without future labels.", "",
        "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m scripts.run_external_validation --stage validate", ".\\.venv\\Scripts\\python.exe -m scripts.run_external_validation --stage test", ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v", "```", "",
        "The paired forecast artifact is `test_predictions.csv.gz` (gzip CSV with timestamp, station, actual, model, prediction). `--stage report` can rebuild descriptive tables and figures from this saved forecast without running a second test. Large local raw data, LightGBM weights, and result files are Git-ignored. Original Favorita artifacts were read only, not modified. Full Favorita and synthetic source tables remain under `results/model_review/`."]
    (OUT / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "test", "report", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("validate", "all"):
        fit_and_lock()
    if args.stage in ("test", "all"):
        evaluate_locked()
    if args.stage == "report":
        postprocess_existing()


if __name__ == "__main__":
    main()
