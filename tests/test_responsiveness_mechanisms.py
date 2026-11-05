"""Independent checks for the pure-information U.S. benchmark."""
import unittest
import numpy as np
from scipy import sparse
from epidemic.responsiveness_mechanisms import checks, objective, allocate
from epidemic.us_proxy_network import loss_gradient


class ResponsivenessMechanismTests(unittest.TestCase):
    def test_weekly_schedule_gradient_and_budget(self):
        n = np.array([1000., 1500., 2000.])
        matrix = sparse.csr_matrix([[0., .6, .4], [.4, 0., .6], [.7, .3, 0.]])
        z = np.random.default_rng(42).normal(0, .3, (12, 3))
        dose = allocate(z, n, .6)
        np.testing.assert_allclose(dose @ n, np.full(12, .6 * n.sum()))
        weekly_beta = np.linspace(.18, .36, 12)
        for mechanism in ['recipient', 'both_ends']:
            args = (n, n * .99, n * .01, np.array([.2, .6, .9]),
                    [matrix] * 12, weekly_beta, .15, mechanism)
            value, grad = objective(dose, *args)
            tangent = dose * (1 - dose / 8)
            derivative = tangent * (grad - n[None, :] * (grad * tangent).sum(axis=1)[:, None] /
                                     (n[None, :] * tangent).sum(axis=1)[:, None])
            for coordinate in [(0, 0), (3, 1), (11, 2)]:
                plus = z.copy(); minus = z.copy()
                plus[coordinate] += 1e-5; minus[coordinate] -= 1e-5
                numerical = (objective(allocate(plus, n, .6), *args, gradient=False) -
                             objective(allocate(minus, n, .6), *args, gradient=False)) / 2e-5
                np.testing.assert_allclose(derivative[coordinate], numerical, rtol=3e-5, atol=1e-5)

    def test_finite_difference_gradients_and_strength_normalization(self):
        checks()

    def test_recipient_mechanism_matches_previous_simulator(self):
        n = np.array([1000., 1500., 2000.])
        matrix = sparse.csr_matrix([[0., .6, .4], [.4, 0., .6], [.7, .3, 0.]])
        matrices = [matrix] * 12
        dose = np.array([.3, .8, .6]); p = np.array([.2, .7, .9])
        value, grad = objective(dose, n, n * .99, n * .01, p, matrices, .3, .15, 'recipient')
        reference, reference_grad = loss_gradient(dose, n, n * .99, n * .01, p, matrices, np.full(12, .3), .15)
        np.testing.assert_allclose(value, reference, rtol=1e-10)
        np.testing.assert_allclose(grad, reference_grad, rtol=1e-10)

    def test_uninformative_response_ensemble_has_identical_objective(self):
        n = np.array([1000., 1500.]); matrix = sparse.csr_matrix([[0., 1.], [1., 0.]])
        for mechanism in ['recipient', 'both_ends']:
            args = (np.array([.4, .6]), n, n * .99, n * .01)
            single = objective(*args, np.full(2, .5), [matrix] * 12, .3, .15, mechanism)
            ensemble = objective(*args, np.full((2, 4), .5), [matrix] * 12, .3, .15, mechanism)
            np.testing.assert_allclose(single[0], ensemble[0])
            np.testing.assert_allclose(single[1], ensemble[1])


if __name__ == '__main__':
    unittest.main()
