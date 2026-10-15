"""Discrete OU process with Gaussian drift and compound-Poisson jump approximation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LevyOU:
    mu: float = 0.0
    rho: float = 0.95
    q: float = 0.01
    p: float = 0.01
    q_jump: float = 0.25

    def __post_init__(self):
        if not (0 <= self.rho < 1 and self.q >= 0 and 0 <= self.p < 1 and self.q_jump >= 0):
            raise ValueError("Require 0<=rho<1, q>=0, 0<=p<1 and q_jump>=0")

    def sample(self, n: int, seed: int = 0, initial: float | None = None):
        if n < 0:
            raise ValueError("n must be nonnegative")
        rng = np.random.default_rng(seed)
        theta = np.empty(n, dtype=float)
        jumps = rng.random(n) < self.p
        state = self.mu if initial is None else float(initial)
        for i in range(n):
            state = self.mu + self.rho * (state - self.mu) + rng.normal(0, np.sqrt(self.q))
            if jumps[i]:
                state += rng.normal(0, np.sqrt(self.q_jump))
            theta[i] = state
        return theta, jumps
