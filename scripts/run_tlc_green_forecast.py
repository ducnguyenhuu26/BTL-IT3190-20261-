"""Locked forecast of daily green-taxi pickup demand by NYC taxi zone."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.collect_tlc_green import CONFIG, PROCESSED, RAW, ROOT, sha256
from src.online_gaussian_bias import KalmanParams, forecast_and_update
from src.tlc_green_data import make_features

OUT = ROOT / "results/tlc_green"
PANEL = PROCESSED / "daily_top_zones.csv.gz"
MODEL = OUT / "lightgbm.txt"
LOCK = OUT / "selection_lock.json"
SOURCE_FILES = {"config": CONFIG, "panel": PANEL,
                "manifest": RAW / "manifest.json",
                "collector": ROOT / "scripts/collect_tlc_green.py",
                "features": ROOT / "src/tlc_green_data.py",
                "filter": ROOT / "src/online_gaussian_bias.py",
                "runner": Path(__file__).resolve()}
FEATURES = ("zone", "dow", "month", "day_of_year", "lag1", "lag7", "lag28",
            "roll7", "roll28")


def load_frame() -> pd.DataFrame:
    quality = json.loads((PROCESSED / "quality_audit.json").read_text(encoding="utf-8"))
    if sha256(PANEL) != quality["panel_sha256"] or sha256(RAW / "manifest.json") != quality["manifest_sha256"]:
        raise ValueError("Processed panel provenance changed")
    frame = pd.read_csv(PANEL, parse_dates=["date"])
    if len(frame) != 12 * 1096 or frame.zone.nunique() != 12:
        raise ValueError("Unexpected daily-zone panel")
    frame = make_features(frame)
    frame["zone"] = pd.Categorical(frame.zone)
    return frame


def splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = frame.date
    return {"train": frame.loc[d < "2024-01-01"],
            "selection": frame.loc[(d >= "2024-01-01") & (d < "2024-07-01")],
            "continuation": frame.loc[(d >= "2024-07-01") & (d < "2025-01-01")],
            "test": frame.loc[d >= "2025-01-01"]}


def fit_model(train: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="regression", n_estimators=300,
                             learning_rate=0.04, num_leaves=31, min_child_samples=30,
                             random_state=42, n_jobs=4, verbosity=-1)
    model.fit(train[list(FEATURES)], np.log1p(train.pickups.to_numpy(float)),
              categorical_feature=["zone"])
    return model


def base_log(model, frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, model.predict(frame[list(FEATURES)], num_threads=4))


def adapted_log(frame: pd.DataFrame, base: np.ndarray, params: KalmanParams) -> np.ndarray:
    output = np.empty(len(frame), dtype=float)
    target = frame.pickups.to_numpy(float)
    for _, positions in frame.groupby("zone", observed=True, sort=False).indices.items():
        pos = np.asarray(positions, dtype=int)
        pos = pos[np.argsort(frame.date.iloc[pos].to_numpy())]
        output[pos] = forecast_and_update(base[pos], target[pos], params)
    return output


def score(frame: pd.DataFrame, prediction_log: np.ndarray) -> dict:
    y = frame.pickups.to_numpy(float)
    z = np.log1p(y)
    return {"n": len(y), "rmsle": float(np.sqrt(np.mean((z-prediction_log)**2))),
            "mae": float(np.mean(np.abs(y-np.expm1(prediction_log))))}


def candidates() -> list[KalmanParams]:
    return [KalmanParams(rho=rho, q=q, observation_variance=r)
            for rho in (0.8, 0.98, 1.0)
            for q in (0.0001, 0.001, 0.01)
            for r in (0.01, 0.1, 1.0)]


def develop() -> None:
    if LOCK.exists():
        raise FileExistsError("Selection already locked")
    parts = splits(load_frame())
    OUT.mkdir(parents=True, exist_ok=True)
    model = fit_model(parts["train"])
    model.booster_.save_model(str(MODEL))
    selection, continuation = parts["selection"], parts["continuation"]
    bs, bc = base_log(model, selection), base_log(model, continuation)
    rows = [{"candidate": "base", "selection_rmsle": score(selection, bs)["rmsle"],
             "continuation_rmsle": score(continuation, bc)["rmsle"]}]
    for i, params in enumerate(candidates()):
        ps, pc = adapted_log(selection, bs, params), adapted_log(continuation, bc, params)
        rows.append({"candidate": str(i), "selection_rmsle": score(selection, ps)["rmsle"],
                     "continuation_rmsle": score(continuation, pc)["rmsle"],
                     "rho": params.rho, "q": params.q,
                     "observation_variance": params.observation_variance})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "development_grid.csv", index=False)
    best = grid.sort_values(["selection_rmsle", "candidate"], kind="stable").iloc[0]
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "minimum Jan-Jun 2024 RMSLE including base; Jul-Dec descriptive only",
            "candidate": str(best.candidate), "params": None if best.candidate == "base" else
            {"rho": float(best.rho), "q": float(best.q),
             "observation_variance": float(best.observation_variance)},
            "split_rows": {k: len(v) for k,v in parts.items()},
            "base_selection_rmsle": float(rows[0]["selection_rmsle"]),
            "selected_selection_rmsle": float(best.selection_rmsle),
            "base_continuation_rmsle": float(rows[0]["continuation_rmsle"]),
            "selected_continuation_rmsle": float(best.continuation_rmsle),
            "sha256": {k: sha256(v) for k,v in {**SOURCE_FILES, "model": MODEL}.items()},
            "seed": 42, "features": list(FEATURES)}
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": lock["candidate"], "params": lock["params"],
                      "selection": [lock["base_selection_rmsle"], lock["selected_selection_rmsle"]],
                      "continuation": [lock["base_continuation_rmsle"], lock["selected_continuation_rmsle"]]},
                     indent=2), flush=True)


def evaluate() -> None:
    if not LOCK.exists():
        raise FileNotFoundError("Run --stage develop first")
    marker = OUT / "test_started.json"
    if marker.exists():
        raise FileExistsError("Locked test already started")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for name,path in {**SOURCE_FILES, "model": MODEL}.items():
        if sha256(path) != lock["sha256"][name]:
            raise ValueError(f"Locked input changed: {name}")
    test = splits(load_frame())["test"]
    base = base_log(lgb.Booster(model_file=str(MODEL)), test)
    selected = base.copy() if lock["params"] is None else adapted_log(
        test, base, KalmanParams(**lock["params"]))
    marker.write_text(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "selection_lock_sha256": sha256(LOCK)}, indent=2), encoding="utf-8")
    results = pd.DataFrame([{"model": "LightGBM", **score(test, base)},
                            {"model": "Selected", **score(test, selected)}])
    results.to_csv(OUT / "test_metrics.csv", index=False)
    pd.DataFrame({"date": test.date, "zone": test.zone.astype(int),
                  "actual": test.pickups, "LightGBM": np.expm1(base),
                  "selected": np.expm1(selected)}).to_csv(
                      OUT / "test_predictions.csv.gz", index=False, compression="gzip")
    print(results.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "test"), required=True)
    args = parser.parse_args()
    if args.stage == "develop":
        develop()
    else:
        evaluate()


if __name__ == "__main__":
    main()
