"""Proposal equations 5–9: weekly exp(-a*p/N) interventions and PPO.

Separate from the earlier capped-adoption simulator. A Gaussian latent vector
parameterizes shared county scores; softmax produces the exact weekly budget.
The joint latent log probability is used in PPO (not a per-county average).
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, softmax
from scipy.stats import binom, t
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class World:
    n: np.ndarray
    s: np.ndarray
    i: np.ndarray
    r: np.ndarray
    p: np.ndarray
    prediction: np.ndarray
    mobility: object
    beta: np.ndarray
    fraction: float
    prior: float
    eligible: object = None
    weeks: int = 12
    effectiveness: float = 1.


def generate_features(count, seed, strength=1.6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, 6))
    true = expit(.8 + strength * (1.2 * x[:, 0] - .9 * x[:, 1] + .6 * np.sin(x[:, 2])))
    return x, true, np.clip(true + rng.normal(0, .06, count), 0, 1)


def predictor():
    x, true, labels = generate_features(4000, 55001)
    learner = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=1).fit(
        x,
        labels)
    xx, yy, _ = generate_features(2000, 55002)
    return learner, float(labels.mean()), {'test_r2': r2_score(yy, learner.predict(xx)),
                                           'test_mae': mean_absolute_error(yy, learner.predict(xx)), 'label_noise_sd': .06}


def synthetic_world(seed, learner, prior, fraction=None, strength=1.6, counties=32):
    rng = np.random.default_rng(seed)
    n = rng.integers(10000, 100000, counties).astype(float)
    infected = np.maximum(1, np.rint(n * rng.uniform(.001, .012, counties)))
    x, p, _ = generate_features(counties, seed + 100000, strength)
    predicted = learner.predict(x)
    neighbors = np.column_stack([(np.arange(counties) + d) %
                                 counties for d in [-2, -1, 1, 2]]).ravel()
    matrix = sparse.coo_matrix((rng.uniform(.2, 1, 4 * counties),
                                (np.repeat(np.arange(counties), 4), neighbors)), shape=(counties, counties)).tocsr()
    matrix = sparse.diags(1 / np.asarray(matrix.sum(1)).ravel()) @ matrix
    rate = rng.uniform(.21, .29)
    beta = np.tile([rate * .85, rate * .15], (12, 1))
    return World(n, n - infected, infected, np.zeros(counties), np.tile(p, (12, 1)),
                 np.tile(predicted, (12, 1)), matrix, beta,
                 float(rng.choice([.02, .10, .20])) if fraction is None else fraction, prior)


class Environment:
    def __init__(self, world, information='blind', seed=0, stochastic=False, paired=False):
        if not np.isfinite(world.effectiveness) or world.effectiveness <= 0:
            raise ValueError('Intervention effectiveness must be positive and finite')
        self.w = world
        self.information = information
        self.s, self.i, self.r = (x.copy() for x in (world.s, world.i, world.r))
        self.week = 0
        self.rng = np.random.default_rng(seed)
        self.stochastic = stochastic
        self.uniforms = self.rng.uniform(
            1e-10, 1 - 1e-10, (world.weeks * 7, 2, len(world.n))) if paired else None
        self.eligible = np.ones(len(world.n), bool) if world.eligible is None else np.asarray(
            world.eligible, bool)
        self.budget = world.fraction * world.n[self.eligible].sum()
        self.total = 0.

    def observed_p(self):
        if self.information == 'blind':
            return np.full(len(self.w.n), self.w.prior)
        if self.information == 'aggregate':
            mean = np.average(self.w.p[self.week, self.eligible], weights=self.w.n[self.eligible])
            return np.full(len(self.w.n), mean)
        if self.information == 'oracle':
            return self.w.p[self.week]
        return self.w.prediction[self.week]

    def matrix(self):
        return self.w.mobility[self.week] if isinstance(self.w.mobility, list) else self.w.mobility

    def features(self):
        if getattr(self, '_feature_cache_week', None) == self.week:
            return self._feature_cache
        n = self.w.n
        pressure = self.i / n
        local, remote = self.w.beta[self.week]
        incoming = self.matrix() @ pressure
        hazard = local * pressure + remote * incoming
        p = self.observed_p()
        log_n = np.clip(np.log(n / np.median(n[self.eligible])), -3, 3)
        log_h = np.clip(np.log(np.maximum(hazard, 1e-8) / .001), -4, 4)
        prevalence = np.clip(pressure * 50, 0, 3)
        susceptible = self.s / n
        central = np.clip(np.log1p(np.asarray(self.matrix().sum(0)).ravel()), 0, 3)
        response = (p - self.w.prior) * 3
        features = np.column_stack([log_n, log_h, prevalence, susceptible - 1,
                                    response, response * log_h, incoming * 50, central, response * susceptible,
                                    np.clip(hazard * susceptible * 500, 0, 3)])
        selected = features[self.eligible]
        context = np.r_[selected.mean(0), selected.std(0), self.week / self.w.weeks,
                        self.w.fraction * 5].astype(np.float32)
        # Base logits express population-proportional spending, common to all
        # information levels. PPO learns the residual county priorities.
        self._feature_cache_week = self.week
        self._feature_cache = (features, context)
        return self._feature_cache

    def allocation(self, weights):
        features, _ = self.features()
        scores = np.log(self.w.n[self.eligible]) + features[self.eligible] @ weights
        allocation = np.zeros(len(self.w.n))
        allocation[self.eligible] = self.budget * softmax(scores)
        return allocation

    def step(self, allocation):
        if np.any(allocation < 0) or not np.isclose(allocation.sum(), self.budget,
                                                    rtol=1e-8) or allocation[~self.eligible].sum() != 0:
            raise ValueError('Weekly budget/eligibility violation')
        # Proposal equation 8. Weekly allocations do not accumulate as a stock.
        multiplier = np.exp(-self.w.effectiveness * allocation * self.w.p[self.week] / self.w.n)
        local, remote = self.w.beta[self.week]
        matrix = self.matrix()
        incidence = np.zeros(len(self.w.n))
        for day in range(7):
            hazard = (local * self.i / self.w.n + remote *
                      (matrix @ (self.i / self.w.n))) * multiplier
            probability = -np.expm1(-hazard)
            recovery = -np.expm1(-1 / 7)
            if self.stochastic:
                if self.uniforms is not None:
                    u = self.uniforms[self.week * 7 + day]
                    new = binom.ppf(u[0], np.rint(self.s).astype(np.int64), probability)
                    recovered = binom.ppf(u[1], np.rint(self.i).astype(np.int64), recovery)
                else:
                    new = self.rng.binomial(np.rint(self.s).astype(np.int64), probability)
                    recovered = self.rng.binomial(np.rint(self.i).astype(np.int64), recovery)
            else:
                new, recovered = self.s * probability, self.i * recovery
            self.s -= new
            self.i += new - recovered
            self.r += recovered
            incidence += new
        self.week += 1
        self.total += incidence.sum()
        return -10 * incidence.sum() / self.w.n.sum(), self.week == self.w.weeks, incidence


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(22, 64), nn.Tanh(), nn.Linear(64, 10))
        self.critic = nn.Sequential(nn.Linear(22, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((10,), -.5))
        nn.init.normal_(self.actor[-1].weight, std=.01)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, obs):
        mean = 3 * torch.tanh(self.actor(obs) / 3)
        return Normal(mean, self.log_std.clamp(-2, .5).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def train(information, learner, prior, seed, steps, destination, world_factory=None, initial_checkpoint=None,
          environment_type=Environment, initialize=None, evaluation_callback=None, actor_penalty=0.):
    torch.manual_seed(seed)
    rng = np.random.default_rng(60000 + seed)
    model = ActorCritic()
    if initial_checkpoint is not None:
        model.load_state_dict(torch.load(initial_checkpoint, weights_only=True))
    if initialize is not None:
        initialize(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    make_world = world_factory if world_factory is not None else lambda world_seed: synthetic_world(
        world_seed, learner, prior)
    environment = environment_type(make_world(int(rng.integers(1, 500000))), information,
                                   seed=seed, stochastic=True)
    history = []
    for update in range((steps + 255) // 256):
        observations, actions, old_logs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(256):
            _, obs = environment.features()
            tensor = torch.as_tensor(obs)
            with torch.no_grad():
                distribution = model.distribution(tensor)
                action = distribution.sample()
                value = model.value(tensor)
                log = distribution.log_prob(action).sum()
            reward, done, _ = environment.step(environment.allocation(action.numpy()))
            observations.append(obs)
            actions.append(action.numpy())
            old_logs.append(log.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            if done:
                history.append({'update': update, 'episode_infections': environment.total,
                                'population': environment.w.n.sum()})
                environment = environment_type(make_world(int(rng.integers(1, 500000))), information,
                                               seed=int(rng.integers(1, 500000)), stochastic=True)
        with torch.no_grad():
            next_value = model.value(torch.as_tensor(environment.features()[1])).item()
        advantages = np.zeros(256, np.float32)
        gae = 0.
        for at in range(255, -1, -1):
            next_v = next_value if at == 255 else values[at + 1]
            delta = rewards[at] + next_v * (1 - dones[at]) - values[at]
            gae = delta + .95 * (1 - dones[at]) * gae  # gamma=1: total infections objective.
            advantages[at] = gae
        returns = torch.as_tensor(advantages + np.asarray(values, np.float32))
        adv = torch.as_tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        obs = torch.as_tensor(np.asarray(observations))
        acts = torch.as_tensor(np.asarray(actions))
        old = torch.as_tensor(old_logs)
        for epoch in range(6):
            for indices in rng.permutation(256).reshape(4, 64):
                distribution = model.distribution(obs[indices])
                logs = distribution.log_prob(acts[indices]).sum(-1)
                ratio = (logs - old[indices]).exp()
                loss_pi = -torch.minimum(ratio * adv[indices],
                                         ratio.clamp(.8, 1.2) * adv[indices]).mean()
                loss_v = .5 * ((model.value(obs[indices]) - returns[indices])**2).mean()
                loss = loss_pi + loss_v - .001 * distribution.entropy().sum(-1).mean()
                loss = loss + actor_penalty * (distribution.mean**2).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
            with torch.no_grad():
                kl = (old - model.distribution(obs).log_prob(acts).sum(-1)).mean().item()
            if kl > .03:
                break
        if evaluation_callback is not None and (update + 1) % 8 == 0:
            evaluation_callback(model, update + 1)
        if (update + 1) % 20 == 0:
            torch.save(model.state_dict(), destination)
            pd.DataFrame(history).to_csv(destination.with_suffix('.training.csv'), index=False)
            print(f'PPO {information} seed{seed}: {(update + 1) * 256}/{steps} weekly decisions', flush=True)
    torch.save(model.state_dict(), destination)
    pd.DataFrame(history).to_csv(destination.with_suffix('.training.csv'), index=False)
    return model


def execute(world, information, policy, seed=0, stochastic=True,
            record=False, environment_type=Environment):
    environment = environment_type(world, information, seed, stochastic, paired=True)
    snapshots = []
    for week in range(world.weeks):
        if isinstance(policy, str):
            p = world.effectiveness * environment.observed_p()
            local, remote = world.beta[week]
            hazard = local * environment.i / world.n + remote * \
                (environment.matrix() @ (environment.i / world.n))
            if policy == 'greedy':
                # Exact water-filling for minimizing instantaneous expected
                # force-weighted exposure sum S*h*exp(-a*p/N).
                slope = environment.s * hazard * p / world.n
                low, high = -60., float(np.log(max(slope.max(), 1e-30)))
                for _ in range(60):
                    log_lagrange = (low + high) / 2
                    action = np.where(
                        p > 0,
                        world.n /
                        np.maximum(
                            p,
                            1e-12) *
                        np.maximum(
                            0,
                            np.log(
                                np.maximum(
                                    slope,
                                    1e-30)) -
                            log_lagrange),
                        0)
                    action[~environment.eligible] = 0
                    if action.sum() > environment.budget:
                        low = log_lagrange
                    else:
                        high = log_lagrange
                action *= environment.budget / max(action.sum(), 1e-30)
            else:
                direction = 1 if policy == 'high_prevalence' else -1
                scores = np.log(world.n) + direction * 4 * environment.i / world.n / .01
                action = np.zeros(len(world.n))
                action[environment.eligible] = environment.budget * \
                    softmax(scores[environment.eligible])
        else:
            with torch.no_grad():
                weights = policy.distribution(torch.as_tensor(
                    environment.features()[1])).mean.numpy()
            action = environment.allocation(weights)
        _, _, new = environment.step(action)
        if record:
            snapshots.append({'week': week + 1, 'incidence': new.copy(), 'allocation': action.copy(),
                              'active': environment.i.copy()})
    return environment.total, snapshots


def summarize(records, output):
    table = pd.DataFrame(records)
    table.to_csv(output / 'trials.csv', index=False)
    results = []
    for keys, group in table.groupby(['scenario', 'fraction']):
        wide = group.pivot(index=['training_seed', 'world'], columns='policy', values='infections')
        # Average training seeds within each independent test world for the CI;
        # do not count three policies on one world as three independent worlds.
        wide = wide.groupby('world').mean()
        for name in wide.columns:
            delta = wide.ppo_blind - wide[name]
            half = t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
            results.append({'scenario': keys[0], 'fraction': keys[1], 'policy': name,
                            'infections': wide[name].mean(), 'gain_vs_blind': delta.mean(),
                            'ci_low': delta.mean() - half, 'ci_high': delta.mean() + half,
                            'relative_percent': 100 * delta.mean() / wide.ppo_blind.mean(), 'independent_worlds': len(delta)})
    pd.DataFrame(results).to_csv(output / 'summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=32768)
    parser.add_argument('--training-seeds', type=int, default=3)
    parser.add_argument('--test-worlds', type=int, default=32)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--test-seed-start', type=int, default=900000)
    parser.add_argument('--output', type=Path, default=Path('outputs/proposal_ppo'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    learner, prior, quality = predictor()
    manifest = {'data_kind': 'FULLY SYNTHETIC PPO MECHANISM EXPERIMENT',
                'proposal_equations': 'exp(-a*p/N) force multiplier; weekly softmax budget; stochastic daily SIR substeps; no effort carryover',
                'architecture': 'Shared 10 county score features; Gaussian 10-dimensional latent scoring coefficients from 22-global-context ->64 tanh->10 actor; critic22->64tanh->1. Identical architecture, steps and settings for blind/oracle/predicted.',
                'steps_per_information_level_per_seed': args.steps, 'training_seeds': args.training_seeds,
                'prior_from_training_labels': prior, 'predictor': quality,
                'training_budgets': [.02, .10, .20], 'test_budgets': [.02, .10, .20],
                'test_world_seed_start': args.test_seed_start, 'test_worlds': args.test_worlds,
                'test_strengths': {'moderate': .7, 'strong': 1.6},
                'uncertainty': 'Paired t intervals over independent test worlds, averaging training seeds within world; exploratory unadjusted; synthetic causality defined by generator',
                'PPO': {'clip': .2, 'gamma': 1., 'gae_lambda': .95, 'epochs': 6, 'rollout': 256, 'minibatch': 64, 'lr': .0003, 'target_kl': .03}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    models = {}
    start = time.monotonic()
    for seed in range(args.seed_start, args.seed_start + args.training_seeds):
        for information in ['blind', 'oracle', 'predicted']:
            path = args.output / f'{information}_{seed}.pt'
            if path.exists():
                history_path = path.with_suffix('.training.csv')
                if not history_path.exists() or pd.read_csv(history_path)[
                        'update'].max() < (args.steps + 255) // 256 - 1:
                    raise ValueError(
                        f'Incomplete training checkpoint: {path}. Preserve it and use a new output folder for a fresh run.')
                model = ActorCritic()
                model.load_state_dict(torch.load(path, weights_only=True))
            else:
                if args.evaluate_only:
                    raise ValueError('Missing completed model ' + str(path))
                model = train(information, learner, prior, seed, args.steps, path)
            models[(seed, information)] = model
    if args.train_only:
        return
    records = []
    for world in range(args.test_worlds):
        for scenario, strength in [('moderate', .7), ('strong', 1.6)]:
            for fraction in [.02, .10, .20]:
                data = synthetic_world(
                    args.test_seed_start +
                    world,
                    learner,
                    prior,
                    fraction,
                    strength)
                for seed in range(args.seed_start, args.seed_start + args.training_seeds):
                    for information in ['blind', 'oracle', 'predicted']:
                        outcome, _ = execute(
                            data, information, models[(seed, information)], 910000 + world)
                        records.append({'scenario': scenario, 'fraction': fraction, 'world': world,
                                        'training_seed': seed, 'policy': 'ppo_' + information, 'infections': outcome})
                for information in ['blind', 'oracle', 'predicted']:
                    outcome, _ = execute(data, information, 'greedy', 910000 + world)
                    # Repeat only for paired table completeness; summary averages
                    # these exact duplicate baselines within a world, not as n.
                    for seed in range(args.seed_start, args.seed_start + args.training_seeds):
                        records.append({'scenario': scenario, 'fraction': fraction, 'world': world,
                                        'training_seed': seed, 'policy': 'greedy_' + information, 'infections': outcome})
        summarize(records, args.output)
        print(f'PPO independent test world {world + 1}/{args.test_worlds}', flush=True)
    print(f'Complete in {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
