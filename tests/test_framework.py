import unittest
import numpy as np
from scipy import sparse
from epidemic.model import Model, Config, State
from epidemic.policies import allocate
from epidemic.experiments import run
from epidemic.data import synthetic


class FrameworkTests(unittest.TestCase):
    def setup_model(self, **kwargs):
        _, n, p, cost, m, initial, _ = synthetic(8)
        model = Model(n, p, cost, m, Config(**kwargs))
        return model, initial

    def test_conservation_and_nonnegative(self):
        for stochastic in (False, True):
            model, state = self.setup_model(stochastic=stochastic)
            rng = np.random.default_rng(1)
            state.effort[:] = 10000
            for _ in range(60):
                u = (rng.random(model.n.shape), rng.random(model.n.shape), rng.random(len(model.n)))
                state, _ = model.step(state, uniforms=u if stochastic else None)
                model.validate_state(state)

    def test_mobility_spreads_infection(self):
        n = np.full((2, 2), 100.)
        infected = np.array([[10., 0.], [0., 0.]])
        state = State(n - infected, infected, np.zeros_like(n), np.zeros(2))
        model = Model(n, [.5, .5], [1, 1], sparse.csr_matrix(
            [[0, 0], [1, 0]]), Config(beta_local=0))
        self.assertGreater(model.incidence(state, model.p)[1].sum(), 0)
        model.cfg.beta_mobility = 0
        self.assertEqual(model.incidence(state, model.p).sum(), 0)

    def test_epidemic_policy_cannot_observe_true_p(self):
        model, state = self.setup_model(budget=100)
        first, _ = allocate(model, state, 100, "epidemic")
        model.p[:] = np.linspace(0, 1, 8)
        second, _ = allocate(model, state, 100, "epidemic")
        np.testing.assert_array_equal(first, second)

    def test_marginal_benefit_cost_and_responsiveness(self):
        n = np.full((2, 2), 1000.)
        i = np.full((2, 2), 10.)
        state = State(n - i, i, np.zeros_like(n), np.zeros(2))
        model = Model(n, [.9, .1], [1, 1], sparse.csr_matrix((2, 2)), Config(chunk=100, budget=100))
        a, _ = allocate(model, state, 100, "oracle")
        self.assertGreater(a[0], a[1])
        model.cost[:] = [100, 1]
        a, _ = allocate(model, state, 100, "oracle")
        self.assertGreater(a[1], a[0])
        gain = model.incidence(state, model.p).sum()
        state.effort[:] = 1000
        self.assertLess(model.incidence(state, model.p).sum(), gain)

    def test_batched_rollout_matches_scalar(self):
        model, state = self.setup_model()
        batch = State(*(np.stack([x, x]) for x in (state.s, state.i, state.r, state.effort)))
        np.testing.assert_allclose(
            model.rollout(
                batch, model.p, 10), model.rollout(
                state, model.p, 10))

    def test_budget_and_reproducibility(self):
        for mode, bound in (("weekly", 300), ("lump", 100)):
            model, state = self.setup_model(
                budget=100, budget_mode=mode, stochastic=True, stochastic_adoption=True)
            a = run(model, state, "oracle", 21, 42)
            b = run(model, state, "oracle", 21, 42)
            self.assertEqual(a[0], b[0])
            self.assertLessEqual(a[0]["spent"], bound + 1e-7)
            self.assertEqual(run(model, state, "none", 21, 42)[0]["spent"], 0)

    def test_network_lookahead_logs_and_budget(self):
        model, state = self.setup_model(budget=50, horizon_days=7)
        a, log = allocate(model, state, 50, "lookahead")
        self.assertLessEqual(a @ model.cost, 50 + 1e-8)
        self.assertTrue(log)
        self.assertTrue(all(row["estimated_gain"] > 0 for row in log))

    def test_time_step_hazard(self):
        model, state = self.setup_model(dt=1)
        probability = model.incidence(state, model.p) / state.s
        model.cfg.dt = 7
        weekly = model.incidence(state, model.p) / state.s
        np.testing.assert_allclose(weekly, 1 - (1 - probability)**7)

    def test_weekly_mobility_and_forecast_information(self):
        model, state = self.setup_model()
        dynamic = Model(
            model.n, model.p, model.cost, {
                0: model.mobility, 7: sparse.csr_matrix(
                    model.mobility.shape)}, model.cfg)
        self.assertGreater(dynamic.mobility_at(0).nnz, 0)
        self.assertEqual(dynamic.mobility_at(7).nnz, 0)
        np.testing.assert_allclose(
            dynamic.rollout(
                state.copy(), model.p, 10), model.rollout(
                state.copy(), model.p, 10))
        with self.assertRaises(ValueError):
            dynamic.mobility_at(14)

    def test_same_eligibility_constraint_across_information_levels(self):
        _, n, p, cost, mobility, state, _ = synthetic(8)
        model = Model(n, p, cost, mobility, Config(budget=100, eligible_counties=(1, 3)))
        for policy in ('epidemic', 'oracle', 'predicted', 'mpc_epidemic', 'mpc'):
            action, _ = allocate(model, state, 100, policy, p)
            self.assertEqual(action[[0, 2, 4, 5, 6, 7]].sum(), 0)

    def test_optional_tensor_parity(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional")
        model, state = self.setup_model()
        batch = State(*(np.stack([x, x]) for x in (state.s, state.i, state.r, state.effort)))
        expected = model.rollout(batch.copy(), model.p, 14)
        model.cfg.rollout_backend = "torch_cpu"
        np.testing.assert_allclose(
            model.rollout(
                batch.copy(),
                model.p,
                14),
            expected,
            rtol=1e-10,
            atol=1e-7)
        if torch.cuda.is_available():
            model.cfg.rollout_backend = "cuda"
            np.testing.assert_allclose(
                model.rollout(
                    batch.copy(),
                    model.p,
                    14),
                expected,
                rtol=1e-9,
                atol=1e-6)


if __name__ == "__main__":
    unittest.main()
