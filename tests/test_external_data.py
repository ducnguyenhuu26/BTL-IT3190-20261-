import copy
import json
from pathlib import Path
import unittest

import pandas as pd

from scripts.collect_external_data import year_queries
from src.external_data import MODEL_FEATURES, make_quake_features, normalize_catalog


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/collection_usgs.json").read_text())


def event(identifier, when, magnitude=5., updated=None):
    stamp = pd.Timestamp(when)
    return {"type": "Feature", "id": identifier,
            "geometry": {"type": "Point", "coordinates": [-79., 0., 15.]},
            "properties": {"time": stamp.value//1000000,
                           "updated": pd.Timestamp(updated or when).value//1000000,
                           "mag": magnitude, "type": "earthquake", "magType": "mw"}}


def collection(*features):
    return {"type": "FeatureCollection", "metadata": {"count": len(features)}, "features": list(features)}


class ExternalDataTests(unittest.TestCase):
    def test_utc_event_after_midnight_can_belong_to_prior_local_day(self):
        result, _ = normalize_catalog([collection(event("a", "2013-01-02T02:00:00Z"))], CONFIG)
        self.assertEqual(result.iloc[0].local_date, pd.Timestamp("2013-01-01"))

    def test_duplicate_event_retains_latest_catalog_revision(self):
        old = event("a", "2013-01-02T12:00:00Z", 5., "2013-01-03T00:00:00Z")
        new = event("a", "2013-01-02T12:00:00Z", 5.2, "2013-02-03T00:00:00Z")
        result, audit = normalize_catalog([collection(new), collection(old)], CONFIG)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].magnitude, 5.2)
        self.assertEqual(audit["duplicate_ids_removed"], 1)

    def test_invalid_response_count_and_coordinates_fail(self):
        payload = collection(event("a", "2013-01-02T12:00:00Z"))
        bad_count = copy.deepcopy(payload)
        bad_count["metadata"]["count"] = 2
        with self.assertRaises(ValueError):
            normalize_catalog([bad_count], CONFIG)
        payload["features"][0]["geometry"]["coordinates"][0] = 20.
        with self.assertRaises(ValueError):
            normalize_catalog([payload], CONFIG)

    def test_real_catalog_schema_can_omit_metadata_count(self):
        empty = {"type": "FeatureCollection", "metadata": {"status": 200}, "features": []}
        populated = collection(event("a", "2013-01-02T12:00:00Z"))
        populated["metadata"].pop("count")
        result, audit = normalize_catalog([empty, populated], CONFIG)
        self.assertEqual(len(result), 1)
        self.assertEqual(audit["raw_response_rows"], 1)

    def test_model_features_do_not_see_same_day_or_future_events(self):
        events, _ = normalize_catalog([collection(event("a", "2013-01-02T12:00:00Z"))], CONFIG)
        changed, _ = normalize_catalog([collection(event("a", "2013-01-02T12:00:00Z"),
                              event("b", "2013-01-03T12:00:00Z", 6.5))], CONFIG)
        before = make_quake_features(events, "2013-01-01", "2013-01-05")
        after = make_quake_features(changed, "2013-01-01", "2013-01-05")
        pd.testing.assert_frame_equal(before.loc[:2, MODEL_FEATURES], after.loc[:2, MODEL_FEATURES])
        self.assertEqual(before.loc[0, "quake_count_prev7d"], 0.)
        self.assertEqual(after.loc[3, "strong_quake_count_lag1"], 1.)
        self.assertTrue(pd.isna(before.loc[0, "quake_max_magnitude"]))

    def test_query_boundaries_include_lookback_and_complete_last_local_day(self):
        queries = year_queries(CONFIG)
        self.assertEqual([year for year, _ in queries], [2012, 2013, 2014, 2015, 2016, 2017])
        self.assertIn("starttime=2012-12-25T05%3A00%3A00", queries[0][1])
        self.assertIn("endtime=2017-08-16T05%3A00%3A00", queries[-1][1])


if __name__ == "__main__":
    unittest.main()
