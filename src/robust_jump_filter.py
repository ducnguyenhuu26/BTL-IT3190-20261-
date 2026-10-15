"""Three-hypothesis moment-matched filter separating latent jumps and outliers."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .levy_process import LevyOU


@dataclass
class RobustFilterState:
    mean: float
    variance: float


class RobustJumpFilter:
    """Approximate Bayesian filter with no-jump, latent-jump, and outlier branches.

    The mutually exclusive branches are H0 (ordinary observation), HJ (latent
    jump, whose posterior is retained), and HO (one-step observation outlier).
    Branch posteriors are moment-matched into one Gaussian state.
    """

    def __init__(
        self,
        process: LevyOU | None = None,
        observation_variance: float = 0.1,
        p_outlier: float = 0.0,
        outlier_variance: float | None = None,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        variance_floor: float = 1e-12,
    ):
        self.process = process or LevyOU()
        self.R = float(observation_variance)
        self.p_outlier = float(p_outlier)
        self.R_out = float(outlier_variance if outlier_variance is not None else 10 * self.R)
        self.prior_mean = float(prior_mean)
        self.prior_variance = float(prior_variance)
        self.floor = float(variance_floor)
        if not math.isfinite(self.R) or self.R <= 0:
            raise ValueError("observation_variance must be finite and positive")
        if not math.isfinite(self.p_outlier) or self.p_outlier < 0:
            raise ValueError("p_outlier must be finite and nonnegative")
        if self.process.p + self.p_outlier >= 1:
            raise ValueError("p_jump + p_outlier must be less than 1")
        if not math.isfinite(self.R_out) or self.R_out <= self.R:
            raise ValueError("outlier_variance must be finite and greater than observation_variance")
        if not math.isfinite(self.prior_variance) or self.prior_variance < 0:
            raise ValueError("prior_variance must be finite and nonnegative")
        if not math.isfinite(self.floor) or self.floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        self.states: dict[object, RobustFilterState] = {}

    def prior(self, key=0) -> tuple[float, float]:
        state = self.states.get(key, RobustFilterState(self.prior_mean, self.prior_variance))
        mean = self.process.mu + self.process.rho * (state.mean - self.process.mu)
        variance = max(self.process.rho**2 * state.variance + self.process.q, self.floor)
        return float(mean), float(variance)

    def prior_many(self, keys):
        """Vectorized priors for one date's independent series states."""
        states = [self.states.get(key) for key in keys]
        means = np.fromiter((state.mean if state is not None else self.prior_mean for state in states), dtype=float, count=len(states))
        variances = np.fromiter((state.variance if state is not None else self.prior_variance for state in states), dtype=float, count=len(states))
        prior_mean = self.process.mu + self.process.rho * (means - self.process.mu)
        prior_variance = np.maximum(self.process.rho**2 * variances + self.process.q, self.floor)
        return prior_mean, prior_variance

    def update_many(self, keys, residuals):
        """Vectorized independent same-date updates; labels never cross series."""
        residuals = np.asarray(residuals, dtype=float)
        if residuals.ndim != 1 or len(keys) != len(residuals) or not np.isfinite(residuals).all():
            raise ValueError("keys/residuals must be aligned 1D values and residuals finite")
        m0, base = self.prior_many(keys)
        y = np.clip(residuals, -1e150, 1e150)
        jump_p, out_p = float(self.process.p), self.p_outlier
        prior_probs = np.asarray([1.0 - jump_p - out_p, jump_p, out_p], dtype=float)
        state_vars = np.stack([base, base + self.process.q_jump, base], axis=1)
        obs_vars = np.asarray([self.R, self.R, self.R_out], dtype=float)
        innovation = y - m0
        total = state_vars + obs_vars[None, :]
        logw = np.full((len(y), 3), -np.inf, dtype=float)
        for branch, probability in enumerate(prior_probs):
            if probability > 0:
                standardized = innovation / np.sqrt(total[:, branch])
                with np.errstate(over="ignore", invalid="ignore"):
                    square = np.square(standardized)
                finite = np.isfinite(square)
                logw[finite, branch] = np.log(probability) - 0.5 * (
                    np.log(2.0 * np.pi * total[finite, branch]) + square[finite]
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
            self.states[key] = RobustFilterState(float(corrected[i]), float(posterior_var[i]))
        return {"prior_mean": m0, "prior_variance": base, "corrected_mean": corrected,
                "variance": posterior_var, "jump_probability": weights[:, 1],
                "outlier_probability": weights[:, 2], "ordinary_probability": weights[:, 0]}

    def predict(self, key=0) -> float:
        return self.prior(key)[0]

    def update(self, key, residual: float) -> dict[str, float]:
        if not np.isfinite(residual):
            raise ValueError("residual must be finite")
        mean_prior, variance_base = self.prior(key)
        # Cap corrupted observations so branch means and moment matching remain
        # representable; likelihoods that underflow fall back to prior weights.
        residual = float(np.clip(residual, -1e150, 1e150))
        jump_p = float(self.process.p)
        out_p = self.p_outlier
        branch_prior = np.asarray([1.0 - jump_p - out_p, jump_p, out_p], dtype=float)
        branch_state_variance = np.asarray(
            [variance_base, variance_base + self.process.q_jump, variance_base], dtype=float
        )
        branch_observation_variance = np.asarray([self.R, self.R, self.R_out], dtype=float)
        innovation = residual - mean_prior
        log_weights = np.full(3, -np.inf, dtype=float)
        posterior_means = np.empty(3, dtype=float)
        posterior_variances = np.empty(3, dtype=float)

        for i, (prior_probability, state_var, obs_var) in enumerate(
            zip(branch_prior, branch_state_variance, branch_observation_variance)
        ):
            innovation_var = max(float(state_var + obs_var), self.floor)
            gain = float(state_var / innovation_var)
            posterior_means[i] = mean_prior + gain * innovation
            posterior_variances[i] = max((1.0 - gain) * state_var, self.floor)
            if prior_probability > 0:
                standardized = innovation / math.sqrt(innovation_var)
                square = standardized * standardized if abs(standardized) < 1e150 else math.inf
                if math.isfinite(square):
                    log_weights[i] = math.log(prior_probability) - 0.5 * (
                        math.log(2.0 * math.pi * innovation_var) + square
                    )

        finite = np.isfinite(log_weights)
        if finite.any():
            peak = float(np.max(log_weights[finite]))
            weights = np.zeros(3, dtype=float)
            weights[finite] = np.exp(log_weights[finite] - peak)
            weights /= weights.sum()
        else:
            weights = branch_prior

        corrected_mean = float(np.dot(weights, posterior_means))
        corrected_variance = float(
            np.dot(weights, posterior_variances + (posterior_means - corrected_mean) ** 2)
        )
        if not math.isfinite(corrected_mean) or not math.isfinite(corrected_variance):
            raise FloatingPointError("non-finite robust filter posterior")
        corrected_variance = max(corrected_variance, self.floor)
        self.states[key] = RobustFilterState(corrected_mean, corrected_variance)
        return {
            "prior_mean": mean_prior,
            "prior_variance": variance_base,
            "corrected_mean": corrected_mean,
            "variance": corrected_variance,
            "jump_probability": float(weights[1]),
            "outlier_probability": float(weights[2]),
            "ordinary_probability": float(weights[0]),
        }
