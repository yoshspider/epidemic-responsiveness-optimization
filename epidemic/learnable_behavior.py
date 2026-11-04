"""Synthetic supervised behavior prediction with untouched test worlds.

This is a mechanism experiment, not an estimate of real mask responsiveness.
Training labels are noisy; interventions execute against latent true p.
"""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from .data import synthetic
from .model import Model, Config
from .experiments import run


def sample(count, seed, strength):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, 6))
    # Only the first three covariates affect response; remaining three are noise.
    p = expit(.9 + strength * (1.2*x[:, 0] - .9*x[:, 1] + .6*np.sin(x[:, 2])))
    labels = np.clip(p + rng.normal(0, .06, count), 0, 1)
    return x, p, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--days', type=int, default=84)
    parser.add_argument('--output', type=Path, default=Path('outputs/learnable_behavior'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design = [('null', 0.), ('moderate', .7), ('strong', 2.)]
    budgets = [.05, .20]
    manifest = {'data_kind': 'FULLY SYNTHETIC, LEARNABLE BEHAVIOR',
        'design': design, 'weekly_budget_fractions': budgets,
        'training_counties': 4000, 'training_seed': 81001,
        'training_label_noise_sd': .06, 'world_seed_start': 82000,
        'feature_seed_start': 83000, 'epidemic_seed_start': 84000,
        'worlds': args.worlds, 'days': args.days,
        'features': 'six independent Gaussian covariates; latent p=expit(0.9+strength*(1.2*x0-0.9*x1+0.6*sin(x2)))',
        'learner': 'RandomForestRegressor(200 trees, min_samples_leaf=8, random_state=17)',
        'uncertainty': 'paired t intervals over independently generated held-out worlds; exploratory, unadjusted',
        'scope': 'Fixed generating mechanisms and budgets retained, including null and losses; no COVID inference'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    trials, prediction = [], []
    for name, strength in design:
        x_train, _, y_train = sample(4000, 81001, strength)
        learner = RandomForestRegressor(n_estimators=200, min_samples_leaf=8,
                                       random_state=17, n_jobs=1).fit(x_train, y_train)
        prior = float(y_train.mean())
        for world in range(args.worlds):
            _, population, _, costs, mobility, initial, _ = synthetic(24, 82000+world)
            x, truth, _ = sample(len(population), 83000+world, strength)
            estimate = np.clip(learner.predict(x), 0, 1)
            if strength == 0:
                # A constant latent response has no county information to exploit.
                # Use the same training-estimated prior for all information policies.
                estimate = np.full(len(truth), prior)
            prediction.extend({'scenario': name, 'world': world, 'county': county,
                               'truth': truth[county], 'prediction': estimate[county],
                               'prior': prior} for county in range(len(truth)))
            for fraction in budgets:
                cfg = Config(prior=prior, mean_prior=prior, stochastic=True,
                             budget=fraction*float(population.sum(1)@costs))
                model = Model(population, truth, costs, mobility, cfg)
                for policy in ('epidemic', 'predicted', 'oracle'):
                    metrics, *_ = run(model, initial, policy, args.days, 84000+world,
                                      estimate if policy == 'predicted' else None, record=False)
                    trials.append({'scenario': name, 'world': world,
                                   'budget_fraction': fraction, **metrics})
            pd.DataFrame(trials).to_csv(args.output/'trials.csv', index=False)
            print(f'{name} test world {world+1}/{args.worlds}', flush=True)
    predictions = pd.DataFrame(prediction)
    predictions.to_csv(args.output/'predictions.csv', index=False)
    prediction_metrics = []
    for name, group in predictions.groupby('scenario'):
        prediction_metrics.append({'scenario': name,
            'mae': mean_absolute_error(group.truth, group.prediction),
            'baseline_mae': mean_absolute_error(group.truth, group.prior),
            'r2': r2_score(group.truth, group.prediction) if group.truth.std()>1e-10 else None,
            'p_mean': group.truth.mean(), 'p_sd': group.truth.std()})
    pd.DataFrame(prediction_metrics).to_csv(args.output/'prediction_metrics.csv', index=False)
    comparisons = []
    for (name, fraction), group in pd.DataFrame(trials).groupby(['scenario', 'budget_fraction']):
        wide = group.pivot(index='world', columns='policy', values='infections')
        oracle_gain = (wide.epidemic-wide.oracle).mean()
        for policy in ('predicted', 'oracle'):
            delta = wide.epidemic-wide[policy]
            half = t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            comparisons.append({'scenario': name, 'budget_fraction': fraction,
                'policy': policy, 'additional_infections_prevented': delta.mean(),
                'ci_low': delta.mean()-half, 'ci_high': delta.mean()+half,
                'relative_percent': 100*delta.mean()/wide.epidemic.mean(),
                'oracle_recovery': delta.mean()/oracle_gain if oracle_gain>0 else None})
    pd.DataFrame(comparisons).to_csv(args.output/'comparisons.csv', index=False)


if __name__ == '__main__':
    main()
