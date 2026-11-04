import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from epidemic.prepare import nyt_daily, ctis_monthly, geods
from epidemic.tbcov import aggregate
from epidemic.calibration import calibrate
from epidemic.data import synthetic
from epidemic.model import Model, Config


class DataTests(unittest.TestCase):
    def test_nyt_revisions_remain_flagged(self):
        frame = pd.DataFrame(
            {"fips": ["01001"] * 3, "date": ["2020-07-01", "2020-07-02", "2020-07-04"], "cases": [10, 8, 12]})
        result, audit = nyt_daily(frame)
        self.assertEqual(result.reported_incidence.tolist(), [10, -2, 4])
        self.assertEqual(audit["negative_revision_rows"], 1)
        self.assertEqual(audit["non_daily_gap_rows"], 1)

    def test_ctis_excludes_state_aggregates(self):
        frame = pd.DataFrame({"geo_value": ["01000", "01001", "01001"], "time_value": [
                             20201001, 20201001, 20201002], "value": [90, 80, 70]})
        result, audit = ctis_monthly(frame)
        self.assertAlmostEqual(result.responsiveness.iloc[0], .75)
        self.assertEqual(audit["state_aggregate_rows_excluded"], 1)

    def test_tbcov_streaming_matches_full_chunk(self):
        frame = pd.DataFrame({"date_time": ["2020-06-01"] * 3, "sentiment_label": [-1, 1, 1], "sentiment_conf": [.9] * 3,
                              "user_loc_country_code": ["US", "US", "CA"], "user_loc_state": ["Alabama"] * 3, "user_loc_county": ["Autauga"] * 3})
        crosswalk = pd.DataFrame({"fips": ["01001"], "state": ["Alabama"], "county": ["Autauga"]})
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "tweets.tsv"
            frame.to_csv(file, sep="\t", index=False)
            small, audit = aggregate(file, crosswalk, chunksize=1)
            large, _ = aggregate(file, crosswalk, chunksize=10)
        pd.testing.assert_frame_equal(small, large)
        self.assertEqual(small.post_count.iloc[0], 2)
        self.assertEqual(small.mean_sentiment.iloc[0], 0)
        self.assertEqual(audit["rows_read"], 3)

    def test_calibration_recovers_known_simulated_rates(self):
        _, n, p, cost, mobility, initial, _ = synthetic(8)
        truth = Model(n, p, cost, mobility, Config(beta_local=.19, beta_mobility=.06))
        state = initial.copy()
        reports = []
        for _ in range(28):
            state, new = truth.step(state)
            reports.append(new.sum(1) * .4)
        guessed = Model(n, p, cost, mobility, Config(beta_local=.22, beta_mobility=.04))
        fit, _ = calibrate(guessed, initial, reports, 20, .4)
        self.assertTrue(fit["success"])
        self.assertAlmostEqual(fit["beta_local"], .19, places=4)
        self.assertAlmostEqual(fit["beta_mobility"], .06, places=4)
        self.assertLess(fit["heldout_rmse"], 1e-3)
