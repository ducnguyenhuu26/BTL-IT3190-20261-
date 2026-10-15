"""Stable moment-matched two-component Bayesian jump-aware filter."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .levy_process import LevyOU


@dataclass
class FilterState:
    mean: float
    variance: float


class JumpAwareFilter:
    def __init__(self, process: LevyOU | None = None, observation_variance: float = 0.1,
                 prior_mean: float = 0.0, prior_variance: float = 1.0, variance_floor: float = 1e-12):
        self.process = process or LevyOU()
        if observation_variance <= 0 or prior_variance < 0 or variance_floor <= 0:
            raise ValueError("Observation variance/floor must be positive; prior variance nonnegative")
        self.R, self.prior_mean = float(observation_variance), float(prior_mean)
        self.prior_variance, self.floor = float(prior_variance), float(variance_floor)
        self.states: dict[object, FilterState] = {}

    def prior(self, key=0):
        st = self.states.get(key, FilterState(self.prior_mean, self.prior_variance))
        m = self.process.mu + self.process.rho * (st.mean - self.process.mu)
        p = max(self.process.rho**2 * st.variance + self.process.q, self.floor)
        return m, p

    def prior_many(self, keys):
        """Vectorized priors for a date batch; keys must be unique within the batch."""
        states = [self.states.get(key) for key in keys]
        means = np.fromiter((state.mean if state is not None else self.prior_mean for state in states), dtype=float, count=len(states))
        variances = np.fromiter((state.variance if state is not None else self.prior_variance for state in states), dtype=float, count=len(states))
        prior_mean = self.process.mu + self.process.rho * (means - self.process.mu)
        prior_variance = np.maximum(self.process.rho**2 * variances + self.process.q, self.floor)
        return prior_mean, prior_variance

    def update_many(self, keys, residuals):
        """Apply independent same-date observations in one vectorized operation."""
        residuals = np.asarray(residuals, dtype=float)
        if residuals.ndim != 1 or len(keys) != len(residuals) or not np.isfinite(residuals).all():
            raise ValueError("keys/residuals must be aligned 1D values and residuals finite")
        m0, base = self.prior_many(keys)
        y = np.clip(residuals, -1e150, 1e150)
        p = float(self.process.p)
        prior_probs = np.asarray([1.0 - p, p])
        state_vars = np.stack([base, base + self.process.q_jump], axis=1)
        innovation = y - m0
        total = state_vars + self.R
        logw = np.full((len(y), 2), -np.inf, dtype=float)
        for branch in range(2):
            if prior_probs[branch] > 0:
                standardized = innovation / np.sqrt(total[:, branch])
                square = np.square(standardized)
                logw[:, branch] = np.log(prior_probs[branch]) - 0.5 * (
                    np.log(2.0 * np.pi * total[:, branch]) + square
                )
        peak = np.max(logw, axis=1)
        weights = np.broadcast_to(prior_probs, logw.shape).copy()
        valid = np.isfinite(peak)
        if valid.any():
            scaled = np.exp(logw[valid] - peak[valid, None])
            weights[valid] = scaled / scaled.sum(axis=1, keepdims=True)
        gains = state_vars / total
        means = m0[:, None] + gains * innovation[:, None]
        variances = np.maximum((1.0 - gains) * state_vars, self.floor)
        corrected = np.sum(weights * means, axis=1)
        posterior_var = np.maximum(np.sum(weights * (variances + np.square(means - corrected[:, None])), axis=1), self.floor)
        for i, key in enumerate(keys):
            self.states[key] = FilterState(float(corrected[i]), float(posterior_var[i]))
        return {"prior_mean": m0, "prior_variance": base, "corrected_mean": corrected,
                "variance": posterior_var, "jump_probability": weights[:, 1]}

    def predict(self, key=0):
        return self.prior(key)[0]

    def update(self, key, residual: float):
        if not np.isfinite(residual):
            raise ValueError("residual must be finite")
        m0, base = self.prior(key)
        # Keep second moments representable even for corrupted/extreme observations.
        residual = float(np.clip(residual, -1e150, 1e150))
        p = self.process.p
        branch_vars = (base, base + self.process.q_jump)
        logw = []
        posts = []
        for k, pk in enumerate(branch_vars):
            s = max(pk + self.R, self.floor)
            innovation = residual - m0
            prior_prob = (1.0 - p) if k == 0 else p
            standardized = innovation / math.sqrt(s)
            squared = standardized * standardized if abs(standardized) < 1e150 else math.inf
            logw.append(-math.inf if prior_prob == 0 else math.log(prior_prob) - 0.5 * (math.log(2 * math.pi * s) + squared))
            gain = pk / s
            mk = m0 + gain * innovation
            postp = max((1 - gain) * pk, self.floor)
            posts.append((mk, postp))
        if p == 0:
            w1 = 0.0
        else:
            norm = np.logaddexp(logw[0], logw[1])
            w1 = float(np.exp(logw[1] - norm)) if np.isfinite(norm) else p
        w0 = 1.0 - w1
        mean = w0 * posts[0][0] + w1 * posts[1][0]
        var = w0 * (posts[0][1] + (posts[0][0] - mean)**2) + w1 * (posts[1][1] + (posts[1][0] - mean)**2)
        self.states[key] = FilterState(float(mean), max(float(var), self.floor))
        return {"prior_mean": m0, "prior_variance": base, "corrected_mean": mean,
                "variance": self.states[key].variance, "jump_probability": w1}
