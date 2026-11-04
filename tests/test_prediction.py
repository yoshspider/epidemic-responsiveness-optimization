import unittest
import pandas as pd
import numpy as np
from epidemic.prediction import predict


class PredictionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"fips": [f"{i + 1:05d}" for i in range(30)],
                             "geo_group": np.repeat(np.arange(5), 6), "feature_end": "2020-06-30",
                             "target_start": "2020-07-02", "responsiveness": np.linspace(.3, .9, 30),
                             "sentiment": np.linspace(-1, 1, 30)})

    def test_out_of_fold_predictions(self):
        predictions, metrics = predict(self.frame(), {"sentiment": ["sentiment"]})
        self.assertEqual(len(metrics), 5)
        self.assertTrue(predictions.sentiment_predicted.between(0, 1).all())

    def test_temporal_and_target_leakage_rejected(self):
        frame = self.frame()
        frame.loc[0, "feature_end"] = "2020-07-03"
        with self.assertRaises(ValueError):
            predict(frame, {"sentiment": ["sentiment"]})
        with self.assertRaises(ValueError):
            predict(self.frame(), {"bad": ["responsiveness"]})

    def test_county_month_temporal_holdout(self):
        early = self.frame()
        early["target_end"] = "2020-07-31"
        late = self.frame()
        late["feature_end"] = "2020-08-31"
        late["target_start"] = "2020-09-01"
        late["target_end"] = "2020-09-30"
        result, metrics = predict(pd.concat([early, late], ignore_index=True), {
                                  "sentiment": ["sentiment"]}, test_start="2020-09-01")
        self.assertTrue(result.sentiment_predicted.iloc[:30].isna().all())
        self.assertTrue(result.sentiment_predicted.iloc[30:].notna().all())
