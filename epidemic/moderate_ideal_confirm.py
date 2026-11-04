"""Fresh-world confirmation after choosing the 12-week beta=.36 lab case."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from .ideal_information_benchmark import outcomes, optimize
from .matched_moderate_benchmark import frontier


def main():
    out = Path('outputs/matched_moderate_benchmark/confirmation')
    out.mkdir(parents=True, exist_ok=True)
    n = 32
    doses = np.arange(0, 8.00001, .1)
    q = (np.arange(1000) + .5) / 1000
    beta = .36
    prior_curve = outcomes(q[:, None], doses[None, :], 84, beta)
    none = float(outcomes([.5], [0.], 84, beta)[0] * n)
    values, _ = frontier(np.tile(prior_curve.mean(0), (n, 1)), round(3 * n / .1))
    budget = int(np.argmin(np.abs((none - values) / none - .4)))
    average = budget * .1 / n
    blind, _ = optimize(np.tile(prior_curve.mean(0), (n, 1)), budget)
    rows = []
    for draw in range(64):
        rng = np.random.default_rng(1003000 + draw)
        p = rng.uniform(0, 1, n)
        cost = outcomes(p[:, None], doses[None, :], 84, beta)
        oracle, _ = optimize(cost, budget)
        choices = {'uninformed': blind, 'perfect_information': oracle}
        for noise in [.05, .15, .3]:
            measured = p + rng.normal(0, noise, n)
            weights = np.exp(-.5 * ((measured[:, None] - q[None, :]) / noise)**2)
            weights /= weights.sum(1)[:, None]
            plan, _ = optimize(weights @ prior_curve, budget)
            choices[f'noise{noise}'] = plan
        for policy, plan in choices.items():
            value = float(cost[np.arange(n), plan].sum())
            rows.append(dict(draw=draw, policy=policy, infections=value,
                        prevention_percent=100 * (none - value) / none))
    pd.DataFrame(rows).to_csv(out / 'trials.csv', index=False)
    wide = pd.DataFrame(rows).pivot(index='draw', columns='policy', values='infections')
    summary = []
    for policy in wide:
        delta = wide.uninformed - wide[policy]
        half = t.ppf(.975, 63) * delta.std(ddof=1) / 8
        prevention = 100 * (none - wide[policy]) / none
        phalf = t.ppf(.975, 63) * prevention.std(ddof=1) / 8
        summary.append(dict(policy=policy, infections=wide[policy].mean(), prevention_percent=prevention.mean(), prevention_ci_low=prevention.mean() - phalf,
                            prevention_ci_high=prevention.mean() + phalf, extra_prevented=delta.mean(), percent_vs_uninformed=100 * delta.mean() / wide.uninformed.mean(), gain_ci_low=delta.mean() - half, gain_ci_high=delta.mean() + half))
    result = pd.DataFrame(summary)
    result.to_csv(out / 'summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(kind='Fresh synthetic confirmation after development regime selection; NOT empirical evidence',
                                                       seeds='1003000–1003063, disjoint from all regime-development response maps', days=84, beta=beta, average_dose=average,
                                                       budget_selection='Frozen from exact global Uniform-prior expected uninformed prevention40%, before held-out response maps',
                                                       population=3200000, counties=32, no_intervention_infections=none, allocator='Globally optimal fixed allocation on grid within independent-county static policy family',
                                                       intervals='95% t over64 random response maps; excludes empirical proxy and model uncertainty'), indent=2))
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
