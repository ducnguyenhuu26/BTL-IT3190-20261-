import unittest

import numpy as np
import pandas as pd

from scripts.build_report import align_predictions, score, table


class ReportBuilderTests(unittest.TestCase):
    def test_paired_alignment_uses_keys_not_file_order(self):
        rows = pd.DataFrame([
            {"date": "2020-01-02", "station": "A", "actual": 3., "prediction": 2., "model": "base"},
            {"date": "2020-01-01", "station": "A", "actual": 1., "prediction": 1., "model": "base"},
            {"date": "2020-01-01", "station": "A", "actual": 1., "prediction": 1.5, "model": "other"},
            {"date": "2020-01-02", "station": "A", "actual": 3., "prediction": 2.5, "model": "other"},
        ])
        wide, n = align_predictions(rows, ["date", "station"], "actual", ["base", "other"])
        self.assertEqual(n, 2)
        self.assertEqual(wide["base"].tolist(), [1., 2.])
        self.assertEqual(wide["other"].tolist(), [1.5, 2.5])

    def test_missing_metric_not_fabricated(self):
        rendered = table(pd.DataFrame({"metric": [np.nan]}))
        self.assertIn("N/A", rendered)
        self.assertAlmostEqual(score(np.array([1., 2.]), np.array([1., 3.]), "mae"), .5)


if __name__ == "__main__":
    unittest.main()
