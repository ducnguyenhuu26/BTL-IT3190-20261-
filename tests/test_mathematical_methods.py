"""Numerical checks of the new convex models and causal online updates."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.hierarchical_residual import BiasPenalties, fit_hierarchical_bias
from src.online_residual import GateConfig, run_gate


class HierarchicalResidualTests(unittest.TestCase):
    def test_eliminated_pair_solution_matches_full_normal_equations(self):
        summary = pd.DataFrame({"store_nbr": [1, 1, 2, 2],
                                "family": ["A", "B", "A", "B"],
                                "sum": [12.0, -2.0, 5.0, 17.0],
                                "count": [5, 8, 6, 7]})
        penalties = BiasPenalties(global_=2.0, store=3.0, family=4.0, pair=5.0)
        fitted = fit_hierarchical_bias(summary, penalties)
        # Direct 1 + 2 stores + 2 families + 4 pairs system is an
        # independent reference for the analytically eliminated solver.
        design = np.zeros((4, 9))
        design[:, 0] = 1.0
        for i, (store, family) in enumerate(zip(summary.store_nbr, summary.family)):
            design[i, store] = 1.0
            design[i, 3 + (family == "B")] = 1.0
            design[i, 5 + i] = 1.0
        count = summary["count"].to_numpy(float)
        target = summary["sum"].to_numpy(float) / count
        ridge = np.diag([2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0])
        direct = np.linalg.solve(design.T @ (count[:, None] * design) + ridge,
                                 design.T @ (count * target))
        expected = design @ direct
        np.testing.assert_allclose(fitted.predict(summary), expected, atol=1e-12)

    def test_invalid_summary_is_rejected(self):
        bad = pd.DataFrame({"store_nbr": [1], "family": ["A"],
                            "sum": [1.0], "count": [0]})
        with self.assertRaises(ValueError):
            fit_hierarchical_bias(bad, BiasPenalties(1, 1, 1, 1))


class OnlineResidualTests(unittest.TestCase):
    def test_same_day_forecasts_cannot_see_current_labels(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-03", "2020-01-02",
                                                      "2020-01-01", "2020-01-02",
                                                      "2020-01-01", "2020-01-03"]),
                              "store_nbr": [1, 1, 1, 2, 2, 2],
                              "family": ["A"] * 6,
                              "actual": [20.0, 30.0, 20.0, 40.0, 30.0, 20.0]})
        base = np.full(6, 12.0)
        config = GateConfig(alpha=.5, method="ogd", eta=.2)
        original = run_gate(frame, base, config)["prediction"]
        changed = frame.copy()
        changed.loc[(changed.date == "2020-01-02") &
                    (changed.store_nbr == 1), "actual"] = 10000.0
        altered = run_gate(changed, base, config)["prediction"]
        same_day = frame.date == "2020-01-02"
        np.testing.assert_allclose(original[same_day], altered[same_day], rtol=0, atol=0)
        self.assertNotEqual(original[0], altered[0])

    def test_ogd_pathwise_regret_bound_against_fixed_gate(self):
        n = 30
        frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n),
                              "store_nbr": np.ones(n, dtype=int),
                              "family": ["A"] * n,
                              "actual": np.tile([12.0, 16.0, 10.0], n // 3)})
        base = np.full(n, 12.0)
        eta = .2
        result = run_gate(frame, base, GateConfig(alpha=.2, eta=eta), return_trace=True)
        y = np.log1p(frame.actual.to_numpy())
        baseline_log = np.log1p(base)
        signal = result["signal"]
        weight = result["weight"]
        learner_loss = np.sum((y - baseline_log - weight * signal) ** 2)
        optimum = np.clip(np.sum(signal * (y - baseline_log)) / np.sum(signal ** 2), 0, 1)
        comparator_loss = np.sum((y - baseline_log - optimum * signal) ** 2)
        # Diameter of [0,1] is 1; eta_t=eta/sqrt(t) is nonincreasing.
        upper_bound = np.sqrt(n) / (2 * eta) + .5 * result["diagnostics"]["step_gradient_sq_sum"]
        self.assertLessEqual(learner_loss - comparator_loss, upper_bound + 1e-10)


if __name__ == "__main__":
    unittest.main()
