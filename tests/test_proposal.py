from dataclasses import replace
import unittest
import numpy as np
from scipy import sparse
from epidemic.proposal_ppo import World, Environment, execute


class ProposalTests(unittest.TestCase):
    def world(self):
        return World(np.array([10000., 10000.]), np.array([9900., 9900.]),
                     np.array([100., 100.]), np.zeros(2), np.tile([.3, .9], (12, 1)),
                     np.tile([.4, .8], (12, 1)), sparse.csr_matrix([[0, 1], [1, 0]]),
                     np.tile([.2, .04], (12, 1)), .1, .6)

    def test_proposal_force_and_conservation(self):
        world = self.world()
        env = Environment(world, 'oracle')
        a = np.array([1000., 1000.])
        s, i, r = world.s.copy(), world.i.copy(), world.r.copy()
        for _ in range(7):
            h = (.2 * i / world.n + .04 * (world.mobility @ (i / world.n))) * \
                np.exp(-a * world.p[0] / world.n)
            new = s * (-np.expm1(-h))
            recovered = i * (-np.expm1(-1 / 7))
            s -= new
            i += new - recovered
            r += recovered
        env.step(a)
        np.testing.assert_allclose(env.s, s)
        np.testing.assert_allclose(env.i, i)
        np.testing.assert_allclose(env.r, r)
        np.testing.assert_allclose(env.s + env.i + env.r, world.n)

    def test_information_mask_and_budget(self):
        world = self.world()
        left = Environment(world, 'blind')
        right = Environment(replace(world, p=1 - world.p), 'blind')
        np.testing.assert_array_equal(left.features()[0], right.features()[0])
        np.testing.assert_array_equal(left.features()[1], right.features()[1])
        action = left.allocation(np.arange(10) / 10)
        self.assertTrue((action >= 0).all())
        self.assertAlmostEqual(action.sum(), left.budget)

    def test_weekly_spending_does_not_accumulate(self):
        env = Environment(self.world(), 'oracle')
        a = env.allocation(np.zeros(10))
        env.step(a)
        self.assertEqual(env.week, 1)
        self.assertFalse(hasattr(env, 'effort'))

    def test_effectiveness_consistent_in_dynamics_and_allocator(self):
        world = self.world()
        scaled = replace(world, effectiveness=3.)
        equivalent = replace(
            world,
            p=world.p * 3,
            prediction=world.prediction * 3,
            prior=world.prior * 3)
        for info in ['blind', 'aggregate', 'oracle', 'predicted']:
            total, left = execute(scaled, info, 'greedy', stochastic=False, record=True)
            other, right = execute(equivalent, info, 'greedy', stochastic=False, record=True)
            self.assertAlmostEqual(total, other, places=7)
            np.testing.assert_allclose([x['allocation'] for x in left],
                                       [x['allocation'] for x in right])

    def test_effectiveness_changes_intervention_not_unfunded_hazard(self):
        world = self.world()
        weak = Environment(world, 'oracle')
        strong = Environment(replace(world, effectiveness=10.), 'oracle')
        action = np.array([1000., 1000.])
        weak.step(action)
        strong.step(action)
        self.assertLess(strong.total, weak.total)
        np.testing.assert_allclose(strong.s + strong.i + strong.r, world.n)
        with self.assertRaises(ValueError):
            Environment(replace(world, effectiveness=0.))
