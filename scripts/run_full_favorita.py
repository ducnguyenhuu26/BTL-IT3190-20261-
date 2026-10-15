"""One locked exploratory full-1782-series Favorita run; never touches earlier results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import load_favorita
from src.features import make_features, feature_columns
from src.jump_filter import JumpAwareFilter
from src.levy_process import LevyOU
from src.robust_jump_filter import RobustJumpFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/full_favorita_exploratory"
MODEL = OUT / "lightgbm_full.txt"
LOCK = OUT / "selected_configs.json"
SOURCE = ROOT / "data/raw/train.csv"
SEED = 42
TRAIN_END = pd.Timestamp("2015-12-31")
VAL_END = pd.Timestamp("2016-03-31")
NAMES = ("Seasonal naive", "LightGBM", "LightGBM + EMA", "LightGBM + Kalman",
         "LightGBM + Robust Kalman", "LightGBM + Jump Filter", "LightGBM + Robust Jump Filter")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(through: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    raw, aux = load_favorita(ROOT / "data/raw")
    if through is not None:
        raw = raw.loc[raw.date <= through]
    if raw[["store_nbr", "family"]].drop_duplicates().shape[0] != 1782:
        raise ValueError("Full Favorita coverage requires 1,782 series")
    frame = make_features(raw, aux)
    del raw
    gc.collect()
    columns = feature_columns(frame)
    # Keep source semantics and all actual-lag features; reduce full-run memory.
    for col in ("sales", "sales_lag1", "sales_lag7", "sales_roll_mean_7",
                "sales_roll_std_7", "sales_roll_mean_28", "sales_roll_std_28"):
        frame[col] = frame[col].astype("float32")
    return frame, columns


def filter_for(family: str, cfg: dict):
    p = cfg.get("p_jump", 0.0)
    process = LevyOU(mu=0, rho=cfg["rho"], q=cfg["q"], p=p, q_jump=cfg.get("q_jump", 0.0))
    if family in ("LightGBM + Kalman", "LightGBM + Jump Filter"):
        return JumpAwareFilter(process, cfg["R"], prior_mean=0, prior_variance=1)
    return RobustJumpFilter(process, cfg["R"], p_outlier=cfg.get("p_outlier", 0.0),
                            outlier_variance=cfg.get("R_out_ratio", 9) * cfg["R"],
                            prior_mean=0, prior_variance=1)


def evaluate(frame: pd.DataFrame, base_log: np.ndarray, family: str, cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or {}
    n = len(frame)
    if len(base_log) != n or not frame.date.is_monotonic_increasing:
        raise ValueError("Frame/base predictions must align and be date sorted")
    if family == "Seasonal naive":
        return np.maximum(0, np.where(np.isfinite(frame.sales_lag7), frame.sales_lag7,
                                      np.where(np.isfinite(frame.sales_lag1), frame.sales_lag1, 0)))
    if family == "LightGBM":
        return np.maximum(0, np.expm1(base_log))
    keys = list(zip(frame.store_nbr.to_numpy(int), frame.family.cat.codes.to_numpy(int)))
    actual = frame.sales.to_numpy(float)
    times = frame.date.to_numpy()
    bounds = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1, n]
    predicted_log = np.empty(n)
    filt = None if family == "LightGBM + EMA" else filter_for(family, cfg)
    ema = {} if filt is None else None
    for begin, end in zip(bounds[:-1], bounds[1:]):
        batch_keys = keys[begin:end]
        prior = (np.fromiter((ema.get(key, 0.) for key in batch_keys), float, count=len(batch_keys))
                 if filt is None else filt.prior_many(batch_keys)[0])
        predicted_log[begin:end] = base_log[begin:end] + prior
        # Every series has a prediction before any same-day target is revealed.
        observed = np.isfinite(actual[begin:end]) & (actual[begin:end] >= 0)
        update_keys = [key for key, ok in zip(batch_keys, observed) if ok]
        residual = np.log1p(actual[begin:end][observed]) - base_log[begin:end][observed]
        if filt is None:
            alpha = cfg["alpha"]
            for key, value in zip(update_keys, residual):
                ema[key] = (1-alpha)*ema.get(key, 0.) + alpha*float(value)
        elif update_keys:
            filt.update_many(update_keys, residual)
    return np.maximum(0, np.expm1(np.clip(predicted_log, -50, 20)))


def score(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    actual = frame.sales.to_numpy(float)
    if len(prediction) != len(actual) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Predictions must cover every row with finite nonnegative values")
    good = np.isfinite(actual) & (actual >= 0)
    a, p = actual[good], prediction[good]
    log_sq = (np.log1p(a) - np.log1p(p))**2
    # Stable compact code: 33 categorical families per store, reused by all models.
    series_code = (frame.loc[good, "store_nbr"].to_numpy(int) * 33 +
                   frame.loc[good, "family"].cat.codes.to_numpy(int))
    counts = np.bincount(series_code)
    used = counts > 0
    series_rmsle = np.sqrt(np.bincount(series_code, weights=log_sq)[used]/counts[used])
    return {"n": int(len(a)), "rmsle": float(np.sqrt(log_sq.mean())),
            "mae": float(np.mean(np.abs(a-p))),
            "mean_series_rmsle": float(series_rmsle.mean())}


def grid(scale: float) -> dict[str, list[dict]]:
    r = max(0.05, float(scale))
    base = [dict(rho=rho, q=q, R=r) for rho in (0.9, 0.98) for q in (0., 0.005)]
    return {
        "LightGBM + EMA": [{"alpha": a} for a in (0.01, 0.05, 0.2)],
        "LightGBM + Kalman": base,
        "LightGBM + Robust Kalman": [dict(x, p_outlier=p, R_out_ratio=25) for x in base for p in (0., .005)],
        "LightGBM + Jump Filter": [dict(x, p_jump=p, q_jump=.3) for x in base for p in (0., .005)],
        "LightGBM + Robust Jump Filter": [dict(x, p_jump=pj, q_jump=.3, p_outlier=po, R_out_ratio=25)
                                          for x in base for pj, po in ((0.,0.),(.005,0.),(0.,.005),(.005,.005))],
    }


def validate() -> None:
    if LOCK.exists():
        raise FileExistsError("Full Favorita config already locked")
    frame, cols = prepare(VAL_END)
    train = frame.loc[frame.date <= TRAIN_END].copy()
    val = frame.loc[frame.date > TRAIN_END].sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    if len(train) != 1945944 or len(val) != 162162:
        raise ValueError("Unexpected full-series train/validation coverage")
    del frame
    gc.collect()
    for col in cols:
        if isinstance(train[col].dtype, pd.CategoricalDtype):
            categories = train[col].cat.categories
            val[col] = val[col].cat.set_categories(categories)
    model = lgb.LGBMRegressor(objective="regression", n_estimators=500,
                              learning_rate=.05, num_leaves=63, random_state=SEED,
                              n_jobs=4, verbosity=-1)
    model.fit(train[cols], np.log1p(train.sales.to_numpy(float)),
              categorical_feature=[col for col in cols if isinstance(train[col].dtype, pd.CategoricalDtype)],
              eval_set=[(val[cols], np.log1p(val.sales.to_numpy(float)))],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"LightGBM full train rows={len(train):,}; validation rows={len(val):,}; best iteration={model.best_iteration_}", flush=True)
    del train
    gc.collect()
    base_log = model.predict(val[cols], num_threads=4)
    residual_var = np.var(np.log1p(val.sales.to_numpy(float)) - base_log)
    rows, selected, val_metrics = [], {}, []
    for family, configs in grid(residual_var).items():
        best = None
        for i, cfg in enumerate(configs):
            pred = evaluate(val, base_log, family, cfg)
            result = score(val, pred)
            rows.append({"model": family, "candidate_id": i, "config": json.dumps(cfg, sort_keys=True), **result})
            rank = (result["rmsle"], i)
            if best is None or rank < best[0]:
                best = (rank, cfg, result)
        selected[family] = best[1]
        val_metrics.append({"model": family, "candidate_count": len(configs), **best[2]})
        print(f"Selected {family}: {best[1]}, validation RMSLE={best[2]['rmsle']:.5f}", flush=True)
    for family in NAMES[:2]:
        val_metrics.append({"model": family, "candidate_count": 0,
                            **score(val, evaluate(val, base_log, family))})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "validation_grid.csv", index=False)
    pd.DataFrame(val_metrics).to_csv(OUT / "validation_metrics.csv", index=False)
    model.booster_.save_model(str(MODEL))
    LOCK.write_text(json.dumps({"locked_at_utc": datetime.now(timezone.utc).isoformat(),
                                "source_sha256": sha256(SOURCE), "model_sha256": sha256(MODEL),
                                "seed": SEED, "best_iteration": int(model.best_iteration_),
                                "features": cols, "selected_configs": selected,
                                "candidate_counts": {k: len(v) for k, v in grid(residual_var).items()},
                                "selection_metric": "validation RMSLE; candidate-order tie break",
                                "scope": "all 1,782 series; exploratory due prior inspection of test period",
                                "state": "zero mean, prior variance 1 at each split start; no warmup"},
                               indent=2), encoding="utf-8")
    print(f"Locked full-run configs at {LOCK}", flush=True)


def bootstrap(wide: pd.DataFrame, repetitions: int = 500) -> pd.DataFrame:
    actual = wide.actual.to_numpy(float)
    block = ((wide.date - wide.date.min()).dt.days // 28).to_numpy(int)
    nblock = int(block.max()) + 1
    counts = np.bincount(block, minlength=nblock)
    sq = {name: np.bincount(block, weights=(np.log1p(actual) -
          np.log1p(wide[name].to_numpy(float)))**2, minlength=nblock) for name in NAMES}
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, nblock, size=(repetitions, nblock))
    denominator = counts[draw].sum(axis=1)
    draws = {name: np.sqrt(values[draw].sum(axis=1)/denominator) for name, values in sq.items()}
    rows = []
    for name in NAMES:
        if name == "LightGBM":
            continue
        refs = ["LightGBM"]
        if name == "LightGBM + Jump Filter":
            refs.append("LightGBM + Kalman")
        if name == "LightGBM + Robust Jump Filter":
            refs.append("LightGBM + Robust Kalman")
        if name == "LightGBM + Robust Kalman":
            refs.append("LightGBM + Kalman")
        for ref in refs:
            delta = draws[name] - draws[ref]
            lo, hi = np.quantile(delta, [0.025, 0.975])
            point = np.sqrt(sq[name].sum()/counts.sum()) - np.sqrt(sq[ref].sum()/counts.sum())
            rows.append({"model": name, "reference": ref, "delta_rmsle": point,
                         "ci_low": lo, "ci_high": hi, "block_days": 28,
                         "blocks": nblock, "bootstrap_reps": repetitions, "seed": SEED})
    return pd.DataFrame(rows)


def report(lock: dict, metrics: pd.DataFrame, intervals: pd.DataFrame) -> str:
    selected = pd.DataFrame([{"model": name, **cfg} for name, cfg in lock["selected_configs"].items()])
    subset = intervals.loc[intervals.reference.isin(["LightGBM", "LightGBM + Kalman",
                                                     "LightGBM + Robust Kalman"])]
    def markdown(data: pd.DataFrame) -> str:
        return data.to_string(index=False, float_format=lambda x: f"{x:.5f}")
    return "\n".join([
        "# Full Favorita exploratory run — all 1,782 series", "",
        "This is a new full-cohort CPU run, separate from Task 01–04 artifacts. The test dates were already inspected in earlier tasks, so these scores are **exploratory**, not a fresh blind confirmation. No result was used to change the locked validation selections.", "",
        f"Source train.csv SHA-256 `{lock['source_sha256']}`; fixed LightGBM SHA-256 `{lock['model_sha256']}`; lock `{lock['locked_at_utc']}`. Seed {SEED}, 4 CPU threads, LightGBM best iteration {lock['best_iteration']}.", "",
        "Train 1,945,944 rows through 2015-12-31; validation 162,162 rows, 2016-01-01–03-31; test 892,782 rows, 2016-04-01–2017-08-15. All 1,782 store×family series are included. Same-day batches are forecast before feedback; filters reset to mean zero, variance one at the start of each split; base LightGBM is fixed during online evaluation.", "",
        "Validation chooses each family on overall RMSLE. Candidate counts are saved in selected_configs.json; all candidates in validation_grid.csv. The one-step lag and rolling features and planned-promotion assumption are the existing Favorita pipeline. Same-day promotion publication time is not verified. Validation is shared with LightGBM early stopping.", "",
        "## Selected configs", "", "```text", markdown(selected), "```", "",
        "## Full exploratory test (sales units for MAE; RMSLE unitless)", "",
        "```text", markdown(metrics), "```", "",
        "## Paired uncertainty", "",
        "28-day time blocks retain every series together; 500 resamples, seed 42. Delta RMSLE is model minus reference; negative favors model. These post-selection intervals describe uncertainty and are not causal proof.", "",
        "```text", markdown(subset), "```", "",
        "Jump filter p=0 configurations are Gaussian limiting cases at matched parameters. Differences among separately selected family configs may also arise from rho/q/R/outlier settings; compare those configs before attributing any gain to jump. Results do not establish physical jump labels.", "",
        "Artifacts: validation_grid.csv, validation_metrics.csv, selected_configs.json, test_metrics.csv, test_predictions.csv.gz, paired_bootstrap.csv. The earlier results under results/baselines, diagnostics, model_review, and external_validation were not overwritten.", "",
    ])


def test_locked() -> None:
    if not LOCK.exists():
        raise FileNotFoundError("Run --stage validate before test")
    if any((OUT / name).exists() for name in ("test_metrics.csv", "test_predictions.csv.gz", "paired_bootstrap.csv")):
        raise FileExistsError("Full Favorita test already evaluated; preserving first result")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if sha256(SOURCE) != lock["source_sha256"] or sha256(MODEL) != lock["model_sha256"]:
        raise ValueError("Source/model differs from locked validation")
    frame, cols = prepare()
    if cols != lock["features"]:
        raise ValueError("Feature list changed after validation lock")
    test = frame.loc[frame.date > VAL_END].sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    del frame
    gc.collect()
    if len(test) != 892782:
        raise ValueError("Unexpected full-series test coverage")
    booster = lgb.Booster(model_file=str(MODEL))
    base_log = booster.predict(test[cols], num_threads=4)
    wide = test[["date", "store_nbr", "family", "sales"]].rename(columns={"sales": "actual"}).copy()
    rows = []
    for family in NAMES:
        pred = evaluate(test, base_log, family, lock["selected_configs"].get(family))
        wide[family] = pred
        result = score(test, pred)
        rows.append({"model": family, **result})
        print(f"Full test {family}: n={result['n']:,}, RMSLE={result['rmsle']:.5f}, MAE={result['mae']:.2f}", flush=True)
    metrics = pd.DataFrame(rows)
    intervals = bootstrap(wide)
    # A completion marker is written last. A failed save cannot be mistaken for
    # a finished run, and prior predictions remain protected from repeat tests.
    wide.to_csv(OUT / "test_predictions.csv.gz", index=False, compression="gzip")
    intervals.to_csv(OUT / "paired_bootstrap.csv", index=False)
    (OUT / "report.md").write_text(report(lock, metrics, intervals), encoding="utf-8")
    metrics.to_csv(OUT / "test_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "test"), required=True)
    args = parser.parse_args()
    if args.stage == "validate":
        validate()
    else:
        test_locked()


if __name__ == "__main__":
    main()
