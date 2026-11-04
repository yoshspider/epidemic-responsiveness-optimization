import unittest
import numpy as np
import pandas as pd
from epidemic.monthly_prediction import fit_predictions


class MonthlyPredictionTests(unittest.TestCase):
    def test_training_target_overlap_is_rejected(self):
        train = pd.DataFrame({'target_end': pd.to_datetime(['2021-01-02']),
                              'responsiveness': [.8], 'previous_mask': [.7]})
        test = pd.DataFrame({'target_start': pd.to_datetime(['2021-01-01']),
                             'previous_mask': [.7]})
        with self.assertRaises(ValueError):
            fit_predictions(train, test, ['previous_mask'], 'ridge', delta=True)

    def test_future_labels_do_not_enter_prediction(self):
        train = pd.DataFrame({'target_end': pd.to_datetime(['2020-12-31'] * 30),
                              'previous_mask': np.linspace(.4, .9, 30),
                              'responsiveness': np.linspace(.42, .92, 30)})
        test = pd.DataFrame({'target_start': pd.to_datetime(['2021-01-01'] * 5),
                             'previous_mask': np.linspace(.5, .8, 5), 'responsiveness': .1})
        first = fit_predictions(train, test, ['previous_mask'], 'ridge', delta=True)
        test['responsiveness'] = .99
        second = fit_predictions(train, test, ['previous_mask'], 'ridge', delta=True)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first, test.previous_mask + .02, atol=1e-10)
