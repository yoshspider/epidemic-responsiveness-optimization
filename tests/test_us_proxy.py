"""Conservation, budget feasibility, and independent numerical gradient checks."""
import unittest
import numpy as np
from scipy import sparse
from epidemic.us_proxy_core import checks as allocation_checks, simulate
from epidemic.us_proxy_network import checks as network_checks


class NationalProxyTests(unittest.TestCase):
    def test_weighted_budget_and_threshold_allocation(self):
        allocation_checks()

    def test_network_adjoint_against_finite_differences(self):
        network_checks()

    def test_zero_responsiveness_removes_intervention_effect(self):
        n = np.array([1000., 2000.])
        i = np.array([10., 20.])
        s = n - i
        r = np.zeros(2)
        matrix = sparse.csr_matrix([[0., 1.], [1., 0.]])
        args = (n, s, i, r, np.zeros(2))
        baseline = simulate(*args, np.zeros(2), [matrix] * 12, np.full(12, .3))
        funded = simulate(*args, np.array([2., .5]), [matrix] * 12, np.full(12, .3))
        np.testing.assert_allclose(baseline['county'], funded['county'])
        self.assertGreater(baseline['total'], 0)


if __name__ == '__main__':
    unittest.main()
