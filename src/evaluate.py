"""Metrics and day-batched online backtests with positional array writes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(actual, predicted, keys=None):
    a = np.asarray(actual, dtype=float)
    p = np.maximum(0, np.asarray(predicted, dtype=float))
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if not len(a):
        return {"n": 0, "rmsle": np.nan, "mae": np.nan, "rmse_log": np.nan, "series_mae_mean": np.nan}
    absolute_error = np.abs(p-a)
    log_error = np.log1p(p)-np.log1p(a)
    log_rmse = float(np.sqrt(np.mean(log_error**2)))
    result = {"n": int(len(a)), "rmsle": log_rmse, "mae": float(absolute_error.mean()),
              "rmse_log": log_rmse, "series_mae_mean": np.nan}
    if keys is not None:
        kept = [key for key, keep in zip(keys, valid) if keep]
        codes, _ = pd.factorize(pd.Index(kept, tupleize_cols=False), sort=False, use_na_sentinel=False)
        sums = np.bincount(codes, weights=absolute_error)
        result["series_mae_mean"] = float(np.mean(sums/np.bincount(codes)))
    return result


def online_adapter_backtest(frame, base_log_prediction, adapter, model_name):
    """Forecast an entire date before feedback; preserve the input row index."""
    out = frame[["date", "store_nbr", "family", "sales"]].copy()
    out["base_log"] = np.asarray(base_log_prediction, dtype=float)
    actual = out.sales.to_numpy()
    base = out.base_log.to_numpy(dtype=float)
    keys = list(zip(out.store_nbr.to_numpy(), out.family.to_numpy()))
    prediction = np.full(len(out), np.nan)
    correction = np.full(len(out), np.nan)
    jump_probability = np.full(len(out), np.nan)
    predict_many = getattr(adapter, "predict_many", None)
    prior_many = getattr(adapter, "prior_many", None)
    update_many = getattr(adapter, "update_many", None)
    for positions in out.groupby("date", sort=True).indices.values():
        batch_keys = [keys[i] for i in positions]
        if callable(predict_many):
            prior = predict_many(batch_keys)
        elif callable(prior_many):
            prior = prior_many(batch_keys)[0]
        else:
            prior = np.fromiter((adapter.predict(key) for key in batch_keys), float, count=len(batch_keys))
        correction[positions] = prior
        prediction[positions] = np.maximum(0, np.expm1(base[positions]+prior))
        residual = np.log1p(actual[positions])-base[positions]
        if callable(update_many):
            state = update_many(batch_keys, residual)
            if isinstance(state, dict) and "jump_probability" in state:
                jump_probability[positions] = state["jump_probability"]
        else:
            for position, key, value in zip(positions, batch_keys, residual):
                state = adapter.update(key, value)
                if isinstance(state, dict) and "jump_probability" in state:
                    jump_probability[position] = state["jump_probability"]
    out["prediction"] = prediction
    out["model"] = model_name
    out["latent_correction"] = correction
    out["jump_probability"] = jump_probability
    out["forecast_error"] = actual-prediction
    return out.rename(columns={"sales": "actual"})[["date", "store_nbr", "family", "actual", "prediction",
           "model", "latent_correction", "jump_probability", "forecast_error"]]
