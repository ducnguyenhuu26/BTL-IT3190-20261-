"""Causal, per-series online calibration of a residual EMA correction.

The base predictor is fixed. Each date is predicted as a batch; targets from
that date only enter the state after all forecasts have been formed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateConfig:
    alpha: float
    method: str = "ogd"
    eta: float = 0.1
    ridge: float = 1.0
    discount: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.method not in {"ogd", "ridge"}:
            raise ValueError("method must be ogd or ridge")
        if self.eta <= 0 or self.ridge <= 0 or not 0 < self.discount <= 1:
            raise ValueError("eta and ridge must be positive; discount in (0, 1]")


def run_gate(
    frame: pd.DataFrame,
    base_sales: np.ndarray,
    config: GateConfig,
    *,
    return_trace: bool = False,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Run a date-batched online gate in the row order of ``frame``.

    ``frame`` needs date, store_nbr, family, actual. One observation per
    (date, store_nbr, family) is required. Rows may be shuffled; chronological
    processing and original output order are both preserved.
    """
    required = {"date", "store_nbr", "family", "actual"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["date", "store_nbr", "family"]).any():
        raise ValueError("Duplicate date and series keys")
    base_sales = np.asarray(base_sales, dtype=float)
    actual = frame.actual.to_numpy(dtype=float)
    if len(base_sales) != len(frame) or not np.isfinite(base_sales).all() or (base_sales < 0).any():
        raise ValueError("base_sales must be finite, nonnegative, and row-aligned")
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("actual must be finite and nonnegative")

    keys = pd.MultiIndex.from_frame(frame[["store_nbr", "family"]])
    codes, unique = pd.factorize(keys, sort=False)
    size = len(unique)
    ema = np.zeros(size, dtype=float)
    weight = np.zeros(size, dtype=float)
    count = np.zeros(size, dtype=np.int32)
    xx = np.zeros(size, dtype=float)
    xy = np.zeros(size, dtype=float)
    base_log = np.log1p(base_sales)
    target_log = np.log1p(actual)
    prediction_log = np.empty(len(frame), dtype=float)
    used_weight = np.empty(len(frame), dtype=float) if return_trace else None
    used_signal = np.empty(len(frame), dtype=float) if return_trace else None
    regret_gradient_sq = 0.0
    regret_step_gradient_sq = 0.0
    for positions in frame.groupby("date", sort=True).indices.values():
        idx = codes[positions]
        signal = ema[idx]
        if config.method == "ridge":
            current_weight = np.clip(xy[idx] / (config.ridge + xx[idx]), 0.0, 1.0)
        else:
            current_weight = weight[idx]
        prediction_log[positions] = np.maximum(0.0, base_log[positions] + current_weight * signal)
        if return_trace:
            used_weight[positions] = current_weight
            used_signal[positions] = signal

        # Feedback is applied only after every forecast in this date batch.
        residual = target_log[positions] - base_log[positions]
        if config.method == "ogd":
            gradient = 2.0 * signal * (current_weight * signal - residual)
            step = config.eta / np.sqrt(count[idx] + 1.0)
            weight[idx] = np.clip(current_weight - step * gradient, 0.0, 1.0)
            regret_gradient_sq += float(np.sum(gradient * gradient))
            regret_step_gradient_sq += float(np.sum(step * gradient * gradient))
        else:
            xx[idx] = config.discount * xx[idx] + signal * signal
            xy[idx] = config.discount * xy[idx] + signal * residual
        ema[idx] = (1.0 - config.alpha) * signal + config.alpha * residual
        count[idx] += 1

    output: dict[str, np.ndarray | dict[str, float]] = {
        "prediction": np.expm1(prediction_log),
        "prediction_log": prediction_log,
    }
    if return_trace:
        output["weight"] = used_weight
        output["signal"] = used_signal
        output["diagnostics"] = {
            "gradient_sq_sum": regret_gradient_sq,
            "step_gradient_sq_sum": regret_step_gradient_sq,
            "series": float(size),
        }
    return output
