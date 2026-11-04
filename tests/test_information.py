import unittest
import json
import numpy as np
from epidemic.exact_information import evaluate


class InformationTests(unittest.TestCase):
    def test_exact_information_value_and_null(self):
        table = evaluate()
        self.assertTrue((table.exact_value_of_information >= -1e-8).all())
        null = table[table.p_values.apply(lambda v: len(set(json.loads(v))) == 1)]
        np.testing.assert_allclose(null.exact_value_of_information, 0, atol=1e-8)
