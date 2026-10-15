"""Gaussian partial pooling for cross-fitted store/family residual bias.

The sufficient statistics are out-of-sample residual sums and counts from a
chronologically earlier calibration period. The solution minimizes a convex
quadratic objective; no test targets enter the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BiasPenalties:
    global_: float
    store: float
    family: float
    pair: float

    def __post_init__(self) -> None:
        if min(self.global_, self.store, self.family, self.pair) <= 0:
            raise ValueError("All ridge penalties must be positive")


@dataclass
class HierarchicalBias:
    global_bias: float
    store_bias: dict[int, float]
    family_bias: dict[str, float]
    pair_bias: dict[tuple[int, str], float]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        stores = frame.store_nbr.to_numpy()
        families = frame.family.to_numpy()
        return np.fromiter(
            (self.global_bias + self.store_bias.get(int(s), 0.0)
             + self.family_bias.get(str(f), 0.0)
             + self.pair_bias.get((int(s), str(f)), 0.0)
             for s, f in zip(stores, families)),
            dtype=float, count=len(frame),
        )


def residual_summary(frame: pd.DataFrame, residual: np.ndarray) -> pd.DataFrame:
    """Aggregate exact sufficient statistics for the Gaussian group model."""
    residual = np.asarray(residual, dtype=float)
    if len(frame) != len(residual) or not np.isfinite(residual).all():
        raise ValueError("Residuals must be finite and row-aligned")
    return (pd.DataFrame({"store_nbr": frame.store_nbr.to_numpy(),
                          "family": frame.family.to_numpy(), "residual": residual})
            .groupby(["store_nbr", "family"], observed=True, sort=True)
            .residual.agg(["sum", "count"]).reset_index())


def fit_hierarchical_bias(summary: pd.DataFrame, penalties: BiasPenalties) -> HierarchicalBias:
    """Solve the exact penalized least-squares normal equations.

    Pair coefficients are eliminated analytically, leaving only a small
    (1 + n_stores + n_families) positive-definite linear system.
    """
    required = {"store_nbr", "family", "sum", "count"}
    if not required.issubset(summary.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(summary.columns))}")
    if summary.duplicated(["store_nbr", "family"]).any():
        raise ValueError("Duplicate store/family summary")
    count = summary["count"].to_numpy(dtype=float)
    total = summary["sum"].to_numpy(dtype=float)
    if len(count) == 0 or not np.isfinite(count).all() or not np.isfinite(total).all() or (count <= 0).any():
        raise ValueError("Counts must be positive and sufficient statistics finite")
    mean = total / count
    stores = sorted(int(v) for v in summary.store_nbr.unique())
    families = sorted(str(v) for v in summary.family.unique())
    store_index = {key: i for i, key in enumerate(stores)}
    family_index = {key: i for i, key in enumerate(families)}
    n = len(summary)
    p = 1 + len(stores) + len(families)
    design = np.zeros((n, p), dtype=float)
    design[:, 0] = 1.0
    for row, (store, family) in enumerate(zip(summary.store_nbr, summary.family)):
        design[row, 1 + store_index[int(store)]] = 1.0
        design[row, 1 + len(stores) + family_index[str(family)]] = 1.0
    # Eliminate pair effect: w_g = n_g/(n_g+lambda_pair)*(mean_g - shared_g).
    effective = count * penalties.pair / (count + penalties.pair)
    ridge = np.r_[penalties.global_,
                  np.full(len(stores), penalties.store),
                  np.full(len(families), penalties.family)]
    gram = design.T @ (effective[:, None] * design)
    gram.flat[::p + 1] += ridge
    shared = np.linalg.solve(gram, design.T @ (effective * mean))
    remainder = count / (count + penalties.pair) * (mean - design @ shared)
    return HierarchicalBias(
        global_bias=float(shared[0]),
        store_bias={store: float(shared[1 + i]) for i, store in enumerate(stores)},
        family_bias={family: float(shared[1 + len(stores) + i]) for i, family in enumerate(families)},
        pair_bias={(int(s), str(f)): float(value) for s, f, value in
                   zip(summary.store_nbr, summary.family, remainder)},
    )
