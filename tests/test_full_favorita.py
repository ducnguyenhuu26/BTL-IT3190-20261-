import unittest

import numpy as np
import pandas as pd

from scripts.run_full_favorita import evaluate, score


class FullFavoritaTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 2 + ["2020-01-03"] * 2),
            "store_nbr": [1, 2] * 3,
            "family": pd.Categorical(["A", "A"] * 3),
            "sales": [2., 4., 3., 5., 4., 6.],
            "sales_lag1": [np.nan, np.nan, 2., 4., 3., 5.],
            "sales_lag7": [np.nan] * 6,
        })
        self.base = np.log1p(np.array([2., 4., 2., 4., 2., 4.]))
        self.cfg = {"rho": .9, "q": .005, "R": .1, "p_jump": 0., "q_jump": .3,
                    "p_outlier": 0., "R_out_ratio": 25}

    def test_same_day_labels_cannot_change_same_day_forecast(self):
        original = evaluate(self.frame, self.base, "LightGBM + Jump Filter", self.cfg)
        changed = self.frame.copy()
        changed.loc[:1, "sales"] = 10000.
        revised = evaluate(changed, self.base, "LightGBM + Jump Filter", self.cfg)
        np.testing.assert_allclose(revised[:2], original[:2])
        self.assertFalse(np.allclose(revised[2:], original[2:]))

    def test_zero_jump_and_outlier_match_gaussian(self):
        plain = evaluate(self.frame, self.base, "LightGBM + Kalman", self.cfg)
        jump = evaluate(self.frame, self.base, "LightGBM + Jump Filter", self.cfg)
        robust = evaluate(self.frame, self.base, "LightGBM + Robust Kalman", self.cfg)
        robust_jump = evaluate(self.frame, self.base, "LightGBM + Robust Jump Filter", self.cfg)
        np.testing.assert_allclose(plain, jump, rtol=0, atol=1e-12)
        np.testing.assert_allclose(plain, robust, rtol=0, atol=1e-12)
        np.testing.assert_allclose(plain, robust_jump, rtol=0, atol=1e-12)

    def test_metric_rejects_missing_prediction(self):
        prediction = np.ones(len(self.frame))
        prediction[0] = np.nan
        with self.assertRaises(ValueError):
            score(self.frame, prediction)


if __name__ == "__main__":
    unittest.main()
