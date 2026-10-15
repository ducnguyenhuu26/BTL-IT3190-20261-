import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.baselines import EMAAdapter, KalmanAdapter
from src.evaluate import metrics, online_adapter_backtest
from src.features import make_features
from src.jump_filter import JumpAwareFilter
from src.levy_process import LevyOU
from src.provenance import archive_code, file_sha256, verify_code


SNAPSHOT = Path(__file__).resolve().parents[1] / "provenance/ede2f98b9ae4"


def reference(name):
    spec = importlib.util.spec_from_file_location("src._original_"+name, SNAPSHOT / "src" / (name+".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OptimizationRegressionTests(unittest.TestCase):
    def test_features_match_original_exactly_with_gaps_missing_values_and_aux(self):
        dates = pd.date_range("2020-01-01", periods=40).delete([2, 9, 31])
        raw = pd.DataFrame([{"date": date, "store_nbr": store, "family": family,
                            "sales": float(i % 7), "onpromotion": i % 2}
                           for store in (1, 2) for family in ("A", "B") for i, date in enumerate(dates)])
        raw.loc[5, "sales"] = np.nan
        raw.sales = raw.sales.astype("float32")
        raw = raw.sample(frac=1, random_state=42)
        aux = {"stores": pd.DataFrame({"store_nbr": [1, 2], "type": ["A", "B"]}),
               "oil": pd.DataFrame({"date": dates, "dcoilwtico": np.arange(len(dates), dtype=float)})}
        expected = reference("features").make_features(raw, aux)
        pd.testing.assert_frame_equal(make_features(raw, aux), expected, check_exact=True)

    def test_batched_backtest_matches_original_scalar_order_and_state(self):
        raw = pd.DataFrame([{"date": date, "store_nbr": store, "family": "A", "sales": float(i+store)}
                            for i, date in enumerate(pd.date_range("2020-01-01", periods=8)) for store in (1, 2, 3)])
        raw.sales = raw.sales.astype("float32")
        raw = raw.sample(frac=1, random_state=42)
        raw.index = np.arange(100, 100+len(raw))
        base = np.log1p(np.full(len(raw), 3.))
        original_eval, original_adapters = reference("evaluate"), reference("baselines")
        process = LevyOU(rho=.9, q=.005, p=.01, q_jump=.3)
        adapters = [(EMAAdapter(.1), original_adapters.EMAAdapter(.1)),
                    (KalmanAdapter(process, .2), original_adapters.KalmanAdapter(process, .2)),
                    (JumpAwareFilter(process, .2), JumpAwareFilter(process, .2))]
        for current, original in adapters:
            with self.subTest(adapter=type(current).__name__):
                expected = original_eval.online_adapter_backtest(raw, base, original, "model")
                got = online_adapter_backtest(raw, base, current, "model")
                pd.testing.assert_frame_equal(got, expected, check_exact=False, atol=1e-11, rtol=1e-12)

    def test_metrics_match_original_with_nonfinite_predictions_and_null_keys(self):
        actual = [0., 2., np.nan, 4., 8., 2.]
        predicted = [-1., 3., 8., np.inf, 5., 4.]
        keys = [(1, "A"), (1, "A"), (2, "A"), (3, "A"), None, None]
        expected = reference("evaluate").metrics(actual, predicted, keys)
        got = metrics(actual, predicted, keys)
        for key in expected:
            self.assertAlmostEqual(got[key], expected[key], places=12)

    def test_historical_snapshot_is_report_only_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"original\r\n")
            expected = {"source.py": file_sha256(source)}
            lock = root / "lock.json"
            lock.write_text(json.dumps({"code_hashes": expected}))
            snapshot = archive_code(root, lock)
            source.write_text("optimized")
            with self.assertRaises(ValueError):
                verify_code(root, lock, expected)
            self.assertEqual(verify_code(root, lock, expected, allow_snapshot=True), "historical_snapshot")
            (snapshot / "source.py").write_text("tampered")
            with self.assertRaises(ValueError):
                verify_code(root, lock, expected, allow_snapshot=True)


if __name__ == "__main__":
    unittest.main()
