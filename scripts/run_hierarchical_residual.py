"""Forward-calibrated hierarchical residual model for the Favorita panel.

Develop fits an auxiliary model on 2013-2014, estimates group residuals on
2015, and selects ridge penalties using Jan-Feb 2016. March is reported as a
chronological confirmation only. Test uses the already-saved base forecasts
and is explicitly exploratory because those dates were inspected before.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.run_full_favorita import MODEL as BASE_MODEL, ROOT, prepare
from src.hierarchical_residual import BiasPenalties, fit_hierarchical_bias, residual_summary

OUT = ROOT / "results/hierarchical_residual_exploratory"
VAL_PRED = ROOT / "results/usgs_ablation_exploratory/validation_predictions.csv.gz"
TEST_PRED = ROOT / "results/usgs_ablation_exploratory/test_predictions.csv.gz"
SOURCE = ROOT / "data/raw/train.csv"
LOCK = OUT / "selected_config.json"
OOF_MODEL = OUT / "calibration_base_2013_2014.txt"
SUMMARY = OUT / "residual_sufficient_statistics_2015.csv"
METHOD_SOURCE = ROOT / "src/hierarchical_residual.py"
SCRIPT_SOURCE = ROOT / "scripts/run_hierarchical_residual.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_forecasts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["date", "store_nbr", "family", "actual", "LightGBM"],
                        parse_dates=["date"])
    if frame.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError(f"Duplicate prediction keys in {path}")
    if frame[["actual", "LightGBM"]].isna().any().any():
        raise ValueError(f"Missing labels or predictions in {path}")
    return frame


def rmsle(actual: np.ndarray, predicted_log: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(actual) - predicted_log) ** 2)))


def candidates() -> list[BiasPenalties]:
    return [BiasPenalties(global_=1000.0, store=store, family=family, pair=pair)
            for store in (100.0, 1000.0, 1e9)
            for family in (100.0, 1000.0, 1e9)
            for pair in (30.0, 100.0, 300.0, 1000.0, 3000.0)]


def develop() -> None:
    if LOCK.exists():
        raise FileExistsError("Development selection already locked")
    OUT.mkdir(parents=True, exist_ok=True)
    frame, features = prepare(pd.Timestamp("2016-03-31"))
    fit_rows = frame.loc[frame.date < "2015-01-01"]
    calibrate = frame.loc[(frame.date >= "2015-01-01") & (frame.date <= "2015-12-31")]
    if len(fit_rows) == 0 or len(calibrate) == 0:
        raise ValueError("Forward calibration periods are empty")
    model = lgb.LGBMRegressor(objective="regression", n_estimators=110,
                              learning_rate=0.05, num_leaves=63,
                              random_state=42, n_jobs=4, verbosity=-1)
    model.fit(fit_rows[features], np.log1p(fit_rows.sales.to_numpy(float)),
              categorical_feature=[name for name in features
                                   if isinstance(fit_rows[name].dtype, pd.CategoricalDtype)])
    model.booster_.save_model(str(OOF_MODEL))
    predicted = np.maximum(0.0, model.predict(calibrate[features], num_threads=4))
    summary = residual_summary(calibrate,
                               np.log1p(calibrate.sales.to_numpy(float)) - predicted)
    summary.to_csv(SUMMARY, index=False, float_format="%.17g")
    validation = read_forecasts(VAL_PRED)
    jan_feb = validation.date < "2016-03-01"
    march = ~jan_feb
    if validation.date.min() != pd.Timestamp("2016-01-01") or validation.date.max() != pd.Timestamp("2016-03-31"):
        raise ValueError("Unexpected validation date span")
    base_log = np.log1p(validation.LightGBM.to_numpy(float))
    actual = validation.actual.to_numpy(float)
    rows = []
    for index, cfg in enumerate(candidates()):
        correction = fit_hierarchical_bias(summary, cfg).predict(validation)
        corrected = np.maximum(0.0, base_log + correction)
        rows.append({"candidate": index, **asdict(cfg),
                     "jan_feb_rmsle": rmsle(actual[jan_feb], corrected[jan_feb]),
                     "march_rmsle": rmsle(actual[march], corrected[march]),
                     "mean_abs_correction": float(np.mean(np.abs(correction)))})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "development_grid.csv", index=False)
    best = grid.sort_values(["jan_feb_rmsle", "candidate"], kind="stable").iloc[0]
    base_jan_feb = rmsle(actual[jan_feb], base_log[jan_feb])
    base_march = rmsle(actual[march], base_log[march])
    selected = {key: float(best[key]) for key in ("global_", "store", "family", "pair")}
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "minimum Jan-Feb 2016 RMSLE; candidate index breaks ties; March is confirmation only",
        "selected_candidate": int(best.candidate),
        "selected_penalties": selected,
        "base_jan_feb_rmsle": base_jan_feb,
        "selected_jan_feb_rmsle": float(best.jan_feb_rmsle),
        "base_march_rmsle": base_march,
        "selected_march_rmsle": float(best.march_rmsle),
        "fit_rows": int(len(fit_rows)),
        "calibration_rows": int(len(calibrate)),
        "source_sha256": sha256(SOURCE),
        "base_model_sha256": sha256(BASE_MODEL),
        "calibration_model_sha256": sha256(OOF_MODEL),
        "summary_sha256": sha256(SUMMARY),
        "validation_predictions_sha256": sha256(VAL_PRED),
        "method_source_sha256": sha256(METHOD_SOURCE),
        "script_source_sha256": sha256(SCRIPT_SOURCE),
        "scope": "Exploratory: prior project stages inspected Favorita test dates",
    }
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "jan_feb": [base_jan_feb, best.jan_feb_rmsle],
                      "march": [base_march, best.march_rmsle]}, indent=2), flush=True)


def block_interval(frame: pd.DataFrame, prediction_log: np.ndarray) -> dict[str, float]:
    y = np.log1p(frame.actual.to_numpy(float))
    base = np.log1p(frame.LightGBM.to_numpy(float))
    blocks = ((frame.date - frame.date.min()).dt.days // 28).to_numpy(int)
    size = int(blocks.max()) + 1
    count = np.bincount(blocks, minlength=size)
    base_sq = np.bincount(blocks, weights=(y - base) ** 2, minlength=size)
    method_sq = np.bincount(blocks, weights=(y - prediction_log) ** 2, minlength=size)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, size, size=(500, size))
    n = count[draws].sum(axis=1)
    delta = np.sqrt(method_sq[draws].sum(axis=1) / n) - np.sqrt(base_sq[draws].sum(axis=1) / n)
    return {"delta_rmsle": float(np.sqrt(method_sq.sum() / count.sum()) -
                                 np.sqrt(base_sq.sum() / count.sum())),
            "ci_low": float(np.quantile(delta, .025)),
            "ci_high": float(np.quantile(delta, .975)),
            "block_days": 28, "blocks": size, "bootstrap_reps": 500}


def test_locked() -> None:
    if not LOCK.exists():
        raise FileNotFoundError("Run --stage develop first")
    if (OUT / "test_started.json").exists():
        raise FileExistsError("This version's test has already started")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    checks = {"source_sha256": SOURCE, "base_model_sha256": BASE_MODEL,
              "calibration_model_sha256": OOF_MODEL, "summary_sha256": SUMMARY,
              "validation_predictions_sha256": VAL_PRED,
              "method_source_sha256": METHOD_SOURCE, "script_source_sha256": SCRIPT_SOURCE}
    for key, path in checks.items():
        if sha256(path) != lock[key]:
            raise ValueError(f"Locked input changed: {key}")
    test_hash = sha256(TEST_PRED)
    test = read_forecasts(TEST_PRED)
    if len(test) != 892782 or test.date.min() != pd.Timestamp("2016-04-01"):
        raise ValueError("Unexpected test coverage")
    summary = pd.read_csv(SUMMARY)
    cfg = BiasPenalties(**lock["selected_penalties"])
    correction = fit_hierarchical_bias(summary, cfg).predict(test)
    prediction_log = np.maximum(0.0, np.log1p(test.LightGBM.to_numpy(float)) + correction)
    prediction = np.expm1(prediction_log)
    marker = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "selected_config_sha256": sha256(LOCK),
              "test_predictions_input_sha256": test_hash}
    (OUT / "test_started.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    metrics = pd.DataFrame([{"model": "LightGBM", "n": len(test),
                             "rmsle": rmsle(test.actual.to_numpy(float),
                                             np.log1p(test.LightGBM.to_numpy(float))),
                             "mae": float(np.mean(np.abs(test.actual - test.LightGBM)))},
                            {"model": "LightGBM + hierarchical residual", "n": len(test),
                             "rmsle": rmsle(test.actual.to_numpy(float), prediction_log),
                             "mae": float(np.mean(np.abs(test.actual - prediction)))}])
    interval = block_interval(test, prediction_log)
    out = test[["date", "store_nbr", "family", "actual", "LightGBM"]].copy()
    out["hierarchical_prediction"] = prediction
    out.to_csv(OUT / "test_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    pd.DataFrame([interval]).to_csv(OUT / "paired_bootstrap.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(interval, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "test"), required=True)
    args = parser.parse_args()
    if args.stage == "develop":
        develop()
    else:
        test_locked()


if __name__ == "__main__":
    main()
