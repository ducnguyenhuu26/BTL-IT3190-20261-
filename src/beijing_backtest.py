"""Hourly online residual-filter evaluation shared by all Beijing branches."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .jump_filter import JumpAwareFilter
from .levy_process import LevyOU
from .robust_jump_filter import RobustJumpFilter


NAMES = ("Seasonal naive", "LightGBM", "LightGBM + EMA", "LightGBM + Kalman",
         "LightGBM + Robust Kalman", "LightGBM + Jump Filter", "LightGBM + Robust Jump Filter")


def make_filter(family: str, config: dict):
    process = LevyOU(mu=0, rho=config["rho"], q=config["q"],
                     p=config.get("p_jump", 0), q_jump=config.get("q_jump", 0))
    if family in ("LightGBM + Kalman", "LightGBM + Jump Filter"):
        return JumpAwareFilter(process, config["R"], prior_mean=0, prior_variance=0)
    return RobustJumpFilter(process, config["R"], p_outlier=config.get("p_outlier", 0),
                            outlier_variance=config.get("R_out_ratio", 9) * config["R"],
                            prior_mean=0, prior_variance=0)


def online_backtest(frame: pd.DataFrame, base_log: np.ndarray, family: str,
                    config: dict | None = None, station_fallback: dict | None = None,
                    diagnostics: bool = False) -> dict[str, np.ndarray]:
    """Predict all stations at an hour, then reveal only finite current labels."""
    config = config or {}
    n = len(frame)
    base_log = np.asarray(base_log, dtype=float)
    if len(base_log) != n or not np.isfinite(base_log).all():
        raise ValueError("One finite base prediction is required for every row")
    if not frame.timestamp.is_monotonic_increasing or frame.duplicated(["timestamp", "station"]).any():
        raise ValueError("Expected unique station rows sorted by timestamp")
    keys = frame.station.astype(str).to_numpy()
    actual = frame["PM2.5"].to_numpy(dtype=float)
    observed = np.isfinite(actual) & (actual >= 0)
    predicted_log = np.full(n, np.nan)
    jump = np.full(n, np.nan) if diagnostics else None
    outlier = np.full(n, np.nan) if diagnostics else None
    if family == "Seasonal naive":
        lag24 = frame.pm_lag24.to_numpy(dtype=float)
        lag1 = frame.pm_lag1.to_numpy(dtype=float)
        fallback = np.array([station_fallback.get(k, 0) for k in keys], dtype=float)
        pred = np.where(np.isfinite(lag24), lag24, np.where(np.isfinite(lag1), lag1, fallback))
        return {"prediction": pred, "prediction_log": np.log1p(np.maximum(pred, 0)),
                "jump_probability": jump, "outlier_probability": outlier}
    if family == "LightGBM":
        pred = np.maximum(0, np.expm1(base_log))
        return {"prediction": pred, "prediction_log": np.log1p(pred),
                "jump_probability": jump, "outlier_probability": outlier}

    filt = None if family == "LightGBM + EMA" else make_filter(family, config)
    ema = {key: 0.0 for key in np.unique(keys)} if filt is None else None
    hours = frame.timestamp.to_numpy()
    bounds = np.r_[0, np.flatnonzero(hours[1:] != hours[:-1]) + 1, n]
    for start, end in zip(bounds[:-1], bounds[1:]):
        batch_keys = keys[start:end].tolist()
        if filt is None:
            correction = np.array([ema[key] for key in batch_keys])
        else:
            correction, _ = filt.prior_many(batch_keys)
        predicted_log[start:end] = base_log[start:end] + correction
        # All forecasts above are fixed before current-hour target updates.
        keep = observed[start:end]
        if not keep.any():
            continue
        update_keys = [key for key, ok in zip(batch_keys, keep) if ok]
        residual = np.log1p(actual[start:end][keep]) - base_log[start:end][keep]
        if filt is None:
            alpha = config["alpha"]
            for key, value in zip(update_keys, residual):
                ema[key] = (1 - alpha) * ema[key] + alpha * float(value)
        else:
            state = filt.update_many(update_keys, residual)
            if diagnostics:
                positions = np.arange(start, end)[keep]
                jump[positions] = state["jump_probability"]
                if "outlier_probability" in state:
                    outlier[positions] = state["outlier_probability"]
    pred = np.maximum(0, np.expm1(np.clip(predicted_log, -50, 20)))
    return {"prediction": pred, "prediction_log": np.log1p(pred),
            "jump_probability": jump, "outlier_probability": outlier}


def score(actual, predicted, station=None) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    good = np.isfinite(actual) & (actual >= 0) & np.isfinite(pred)
    a, p = actual[good], np.maximum(0, pred[good])
    if not len(a):
        raise ValueError("No scored target rows")
    result = {"n": int(len(a)), "rmse": float(np.sqrt(np.mean((p-a)**2))),
              "mae": float(np.mean(np.abs(p-a))),
              "rmse_log": float(np.sqrt(np.mean((np.log1p(p)-np.log1p(a))**2)))}
    if station is not None:
        data = pd.DataFrame({"station": np.asarray(station)[good], "actual": a, "prediction": p})
        per = data.groupby("station", observed=True).apply(
            lambda x: pd.Series({"rmse": np.sqrt(np.mean((x.prediction-x.actual)**2)),
                                 "mae": np.mean(np.abs(x.prediction-x.actual)),
                                 "rmse_log": np.sqrt(np.mean((np.log1p(x.prediction)-np.log1p(x.actual))**2))}),
            include_groups=False)
        result.update({f"mean_station_{col}": float(per[col].mean()) for col in per})
    return result
