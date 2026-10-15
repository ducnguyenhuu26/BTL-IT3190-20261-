from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.run_usgs_ablation import best_blend, test as run_locked_test
from src.external_data import MODEL_FEATURES, join_quake_features


class USGSAblationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"date": pd.to_datetime(["2013-01-02", "2013-01-01", "2013-01-02"]),
                                   "store_nbr": [1, 1, 2], "sales": [2., 3., 4.]})
        self.calendar = pd.DataFrame({"date": pd.to_datetime(["2013-01-01", "2013-01-02"]),
             "quake_count_lag1": [0., 1.], "strong_quake_count_lag1": [0., 0.],
             "quake_max_magnitude_lag1": [np.nan, 5.], "quake_count_prev7d": [0., 1.],
             "quake_count": [1000., 2000.], "sales_sum": [99999., 99999.]})

    def test_join_uses_only_allowlisted_features_and_preserves_sales_rows(self):
        result = join_quake_features(self.frame, self.calendar)
        pd.testing.assert_frame_equal(result[self.frame.columns], self.frame)
        self.assertEqual(result.quake_count_lag1.tolist(), [1., 0., 1.])
        self.assertNotIn("quake_count", result)
        self.assertNotIn("sales_sum", result)
        self.assertEqual(set(result.columns)-set(self.frame.columns), set(MODEL_FEATURES))

    def test_join_rejects_duplicate_or_missing_calendar_dates(self):
        with self.assertRaises(ValueError):
            join_quake_features(self.frame, pd.concat([self.calendar, self.calendar.iloc[:1]]))
        with self.assertRaises(ValueError):
            join_quake_features(self.frame, self.calendar.iloc[:1])

    def test_blend_closed_form_and_constraint_boundaries(self):
        a, b = np.zeros(5), np.full(5, 4.)
        self.assertAlmostEqual(best_blend(np.ones(5), a, b), .25)
        self.assertEqual(best_blend(np.full(5, -1.), a, b), 0.)
        self.assertEqual(best_blend(np.full(5, 6.), a, b), 1.)
        self.assertEqual(best_blend(np.ones(5), a, a), 0.)

    def test_blend_fails_on_nonfinite_or_misaligned_predictions(self):
        with self.assertRaises(ValueError):
            best_blend(np.ones(2), np.ones(3), np.ones(2))
        with self.assertRaises(ValueError):
            best_blend(np.ones(2), np.array([np.nan, 1.]), np.ones(2))

    def test_test_marker_prevents_any_repeat_model_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "test_started.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                run_locked_test({}, output)


if __name__ == "__main__":
    unittest.main()
