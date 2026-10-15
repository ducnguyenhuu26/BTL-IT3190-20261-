"""Online residual adapters; caller must predict a complete date before updating."""
from __future__ import annotations
import numpy as np
from .jump_filter import JumpAwareFilter
from .levy_process import LevyOU


class EMAAdapter:
    def __init__(self, alpha=0.1):
        if not 0 < alpha <= 1: raise ValueError("alpha must be in (0,1]")
        self.alpha, self.state = alpha, {}
    def predict(self, key): return self.state.get(key, 0.0)
    def update(self, key, residual): self.state[key] = (1-self.alpha)*self.predict(key) + self.alpha*float(residual)


class KalmanAdapter:
    def __init__(self, process: LevyOU, observation_variance=0.1, prior_variance=1.0):
        self.filter = JumpAwareFilter(LevyOU(process.mu, process.rho, process.q, 0, 0), observation_variance, process.mu, prior_variance)
    def predict(self,key): return self.filter.prior(key)[0]
    def update(self,key,residual): return self.filter.update(key,residual)


def seasonal_naive(history: dict, key, date, fallback=0.0):
    return history.get((key, date - np.timedelta64(7, "D")), fallback)
