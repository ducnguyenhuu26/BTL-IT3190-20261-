"""Metrics and day-batched online backtest utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd


def metrics(actual, predicted, keys=None):
    a, p = np.asarray(actual, float), np.maximum(0, np.asarray(predicted, float))
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a): return {"n":0,"rmsle":np.nan,"mae":np.nan,"rmse_log":np.nan,"series_mae_mean":np.nan}
    out={"n":int(len(a)),"rmsle":float(np.sqrt(np.mean((np.log1p(p)-np.log1p(a))**2))),
         "mae":float(np.mean(np.abs(p-a))),"rmse_log":float(np.sqrt(np.mean((np.log1p(p)-np.log1p(a))**2)))}
    if keys is not None:
        kept=[key for key, keep in zip(keys, ok) if keep]
        tmp=pd.DataFrame({"key":kept,"err":np.abs(p-a)})
        out["series_mae_mean"] = float(tmp.groupby("key",dropna=False).err.mean().mean())
    else: out["series_mae_mean"] = np.nan
    return out


def online_adapter_backtest(frame, base_log_prediction, adapter, model_name):
    """Predict all rows on each date, then update per-series state from actual labels."""
    out=frame[["date","store_nbr","family","sales"]].copy()
    out["base_log"] = np.asarray(base_log_prediction, float)
    out["prediction"] = np.nan
    out["model"] = model_name
    out["latent_correction"] = np.nan
    out["jump_probability"] = np.nan
    for date, idx in out.groupby("date", sort=True).groups.items():
        rows=out.loc[idx]
        corr=np.array([adapter.predict((r.store_nbr,r.family)) for r in rows.itertuples()])
        out.loc[idx,"prediction"] = np.maximum(0,np.expm1(rows.base_log.to_numpy()+corr))
        out.loc[idx,"latent_correction"] = corr
        # Update only after every forecast for this date has been formed.
        residual=np.log1p(rows.sales.to_numpy())-rows.base_log.to_numpy()
        for row, res in zip(rows.itertuples(), residual):
            state=adapter.update((row.store_nbr,row.family),res)
            if isinstance(state, dict) and "jump_probability" in state:
                out.loc[row.Index,"jump_probability"] = state["jump_probability"]
    out["forecast_error"] = out.sales-out.prediction
    return out.rename(columns={"sales":"actual"})[["date","store_nbr","family","actual","prediction","model","latent_correction","jump_probability","forecast_error"]]
