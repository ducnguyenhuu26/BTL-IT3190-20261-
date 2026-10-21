"""One-step Gaussian state-space correction for a fixed base forecast."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanParams:
    rho: float
    q: float
    observation_variance: float

    def __post_init__(self) -> None:
        if not 0 <= self.rho <= 1 or self.q < 0 or self.observation_variance <= 0:
            raise ValueError("Invalid state-space parameters")


def forecast_and_update(base_log: np.ndarray, actual: np.ndarray,
                        params: KalmanParams) -> np.ndarray:
    """Predict at t from state at t-1, then consume only label t.

    Under the linear Gaussian model, the prior mean is the minimum-MSE
    conditional forecast correction. Each call starts at prior N(0, 0).
    """
    base_log = np.asarray(base_log, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if base_log.shape != actual.shape or not np.isfinite(base_log).all():
        raise ValueError("Aligned finite base predictions are required")
    prediction = np.empty_like(base_log)
    mean = variance = 0.0
    for i, (base, y) in enumerate(zip(base_log, actual)):
        prior_mean = params.rho * mean
        prior_var = params.rho**2 * variance + params.q
        prediction[i] = max(0.0, base + prior_mean)
        if np.isfinite(y) and y >= 0:
            residual = np.log1p(y) - base
            gain = prior_var / (prior_var + params.observation_variance)
            mean = prior_mean + gain * (residual - prior_mean)
            variance = (1.0 - gain) * prior_var
        else:
            mean, variance = prior_mean, prior_var
    return prediction
