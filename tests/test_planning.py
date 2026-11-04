import unittest
import numpy as np
from epidemic.data import synthetic
from epidemic.model import Config, Model
from epidemic.planning import objective, objective_gradient, plan
from epidemic.policies import allocate


class PlanningTests(unittest.TestCase):
    def model(self, mode="weekly"):
        _, n, p, cost, mobility, state, _ = synthetic(6)
        return Model(n, p, cost, mobility, Config(budget=100, horizon_days=14,
                     planning_iterations=4, budget_mode=mode)), state

    def test_adjoint_matches_finite_difference(self):
        model, state = self.model()
        controls = np.zeros((14, 6))
        controls[0] = 10
        controls[7] = 5
        value, gradient = objective_gradient(model, state, model.p, controls, 14)
        self.assertAlmostEqual(value, objective(model, state, model.p, controls, 14), places=7)
        for row, col in ((0, 0), (0, 3), (7, 2), (13, 5)):
            plus = controls.copy()
            minus = controls.copy()
            plus[row, col] += .001
            minus[row, col] -= .001
            difference = (objective(model, state, model.p, plus, 14) -
                          objective(model, state, model.p, minus, 14)) / .002
            self.assertAlmostEqual(gradient[row, col], difference, places=5)

    def test_mpc_budget_and_descent(self):
        for mode in ("weekly", "lump"):
            model, state = self.model(mode)
            action, logs = plan(model, state, 100, model.p)
            self.assertLessEqual(action @ model.cost, 100 + 1e-7)
            self.assertTrue(np.all(action >= 0))
            if logs:
                self.assertLessEqual(logs[0]["forecast_after"], logs[0]["forecast_before"] + 1e-7)

    def test_mpc_blind_information_isolation(self):
        model, state = self.model()
        a, _ = allocate(model, state, 100, "mpc_epidemic")
        model.p[:] = 0
        b, _ = allocate(model, state, 100, "mpc_epidemic")
        np.testing.assert_array_equal(a, b)

    def test_adjoint_uses_current_transmission_schedule(self):
        model, state = self.model()
        model.cfg.beta_schedule = ((0, .08, .02), (7, .25, .07))
        state.day = 7
        controls = np.zeros((14, 6))
        controls[0] = 10
        _, gradient = objective_gradient(model, state, model.p, controls, 14)
        plus, minus = controls.copy(), controls.copy()
        plus[0, 2] += .001
        minus[0, 2] -= .001
        difference = (objective(model, state, model.p, plus, 14) -
                      objective(model, state, model.p, minus, 14)) / .002
        self.assertAlmostEqual(gradient[0, 2], difference, places=5)
