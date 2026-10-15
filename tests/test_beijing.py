import unittest
import numpy as np
import pandas as pd

from src.beijing_data import END, START, MEASURES, load_beijing, make_beijing_features, split_beijing
from src.beijing_backtest import online_backtest, make_filter
from scripts.run_external_validation import RAW, candidates, paired_bootstrap


def miniature():
    times = pd.to_datetime(["2016-06-30 22:00", "2016-07-01 00:00", "2016-07-01 01:00"])
    rows = []
    for station, offset in (("A", 0), ("B", 10)):
        for i, timestamp in enumerate(times):
            row = {"station": station, "timestamp": timestamp, "wd": "N",
                   **{col: float(i + offset) for col in MEASURES}}
            rows.append(row)
    return pd.DataFrame(rows)


class BeijingTests(unittest.TestCase):
    def test_real_coverage_and_disjoint_split(self):
        if not RAW.exists():
            self.skipTest("Optional real Beijing data is not present; install it to run this integration check")
        raw = load_beijing(RAW)
        self.assertEqual(len(raw), 420768)
        self.assertEqual(raw.station.nunique(), 12)
        self.assertEqual(raw.timestamp.min(), START)
        self.assertEqual(raw.timestamp.max(), END - pd.Timedelta(hours=1))
        split = split_beijing(raw)
        self.assertEqual(sum(len(x) for x in split.values()), len(raw))
        self.assertLess(split["train"].timestamp.max(), split["validation"].timestamp.min())
        self.assertLess(split["validation"].timestamp.max(), split["test"].timestamp.min())

    def test_exact_hour_lags_and_causality(self):
        raw = miniature()
        a, _ = make_beijing_features(raw)
        row = a.loc[(a.station == "A") & (a.timestamp == pd.Timestamp("2016-07-01 00:00"))].iloc[0]
        self.assertTrue(np.isnan(row.pm_lag1))  # Missing 23:00 is not 22:00.
        next_row = a.loc[(a.station == "A") & (a.timestamp == pd.Timestamp("2016-07-01 01:00"))].iloc[0]
        self.assertEqual(next_row.pm_lag3, 0)
        self.assertEqual(row.pm_roll24_mean, 0)
        changed = raw.copy()
        changed.loc[changed.timestamp >= pd.Timestamp("2016-07-01 00:00"), ["PM2.5", "NO2"]] = 999
        b, columns = make_beijing_features(changed)
        index = (a.station == "A") & (a.timestamp == pd.Timestamp("2016-07-01 00:00"))
        pd.testing.assert_series_equal(a.loc[index, columns].iloc[0], b.loc[index, columns].iloc[0])
        self.assertNotEqual(a.loc[a.timestamp == pd.Timestamp("2016-07-01 01:00"), "pm_lag1"].iloc[0],
                            b.loc[b.timestamp == pd.Timestamp("2016-07-01 01:00"), "pm_lag1"].iloc[0])

    def test_current_and_other_station_targets_cannot_change_forecast(self):
        raw = miniature()
        a, _ = make_beijing_features(raw)
        a = a.loc[a.timestamp >= pd.Timestamp("2016-07-01")].sort_values(["timestamp", "station"]).reset_index(drop=True)
        first = online_backtest(a, np.ones(len(a)), "LightGBM + EMA", {"alpha": 0.5})
        changed = a.copy()
        changed.loc[changed.timestamp == pd.Timestamp("2016-07-01"), "PM2.5"] = 900
        second = online_backtest(changed, np.ones(len(a)), "LightGBM + EMA", {"alpha": 0.5})
        np.testing.assert_array_equal(first["prediction"][:2], second["prediction"][:2])
        self.assertFalse(np.array_equal(first["prediction"][2:], second["prediction"][2:]))

    def test_missing_target_skips_update_and_metric(self):
        a, _ = make_beijing_features(miniature())
        a = a.loc[a.timestamp >= pd.Timestamp("2016-07-01")].sort_values(["timestamp", "station"]).reset_index(drop=True)
        x = a.copy(); x.loc[0, "PM2.5"] = np.nan
        y = a.copy(); y.loc[0, "PM2.5"] = np.nan
        first = online_backtest(x, np.ones(len(x)), "LightGBM + EMA", {"alpha": .5})
        second = online_backtest(y, np.ones(len(y)), "LightGBM + EMA", {"alpha": .5})
        np.testing.assert_array_equal(first["prediction"], second["prediction"])
        self.assertAlmostEqual(first["prediction"][0], first["prediction"][2])

    def test_filter_limiting_cases_and_seed(self):
        cfg = dict(rho=.9, q=.01, R=.2, p_jump=0, q_jump=.5,
                   p_outlier=0, R_out_ratio=25)
        kalman = make_filter("LightGBM + Kalman", cfg)
        robust = make_filter("LightGBM + Robust Jump Filter", cfg)
        for batch in ([.2, -.5], [1.2, .3]):
            a = kalman.update_many(["A", "B"], batch)
            b = robust.update_many(["A", "B"], batch)
            np.testing.assert_allclose(a["corrected_mean"], b["corrected_mean"], atol=1e-12)
        self.assertEqual(candidates(.2), candidates(.2))
        frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=168, freq="h"),
                              "actual": np.arange(168, dtype=float),
                              **{name: np.arange(168, dtype=float) + i for i, name in
                                 enumerate(("Seasonal naive", "LightGBM", "LightGBM + EMA",
                                            "LightGBM + Kalman", "LightGBM + Robust Kalman",
                                            "LightGBM + Jump Filter", "LightGBM + Robust Jump Filter"))}})
        pd.testing.assert_frame_equal(paired_bootstrap(frame, 10), paired_bootstrap(frame, 10))


if __name__ == "__main__":
    unittest.main()
