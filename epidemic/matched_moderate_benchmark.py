"""Budget matched to 40% uninformed prevention, before test response labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from .ideal_information_benchmark import outcomes, optimize


def frontier(cost, budget):
    counties, choices = cost.shape
    value = np.full(budget + 1, np.inf)
    value[0] = 0
    decisions = np.full((counties, budget + 1), -1, dtype=np.int16)
    for county in range(counties):
        new = np.full_like(value, np.inf)
        for dose in range(min(choices - 1, budget) + 1):
            candidate = value[:budget + 1 - dose] + cost[county, dose]
            take = candidate < new[dose:]
            new[dose:][take] = candidate[take]
            decisions[county, dose:][take] = dose
        value = new
    return value, decisions


def main():
    out = Path('outputs/matched_moderate_benchmark')
    out.mkdir(parents=True, exist_ok=True)
    n = 32
    doses = np.arange(0, 8.00001, .1)
    q = (np.arange(1000) + .5) / 1000
    records = []
    calibration = []
    for days in [84, 168]:
        for beta in [.24, .30, .36, .42]:
            prior_curve = outcomes(q[:, None], doses[None, :], days, beta)
            none = float(outcomes([.5], [0.], days, beta)[0] * n)
            costs = np.tile(prior_curve.mean(0), (n, 1))
            values, decisions = frontier(costs, round(3 * n / .1))
            budget = int(np.argmin(np.abs((none - values) / none - .4)))
            blind = np.zeros(n, dtype=int)
            remaining = budget
            for county in reversed(range(n)):
                blind[county] = decisions[county, remaining]
                remaining -= blind[county]
            if remaining or blind.sum() != budget:
                raise ValueError('Reference budget mismatch')
            calibration.append(dict(days=days, beta=beta, average_dose=budget * .1 / n,
                               expected_uninformed_prevention_percent=100 * (none - values[budget]) / none))
            for draw in range(24):
                rng = np.random.default_rng(1001000 + draw)
                p = rng.uniform(0, 1, n)
                cost = outcomes(p[:, None], doses[None, :], days, beta)
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
                    records.append(dict(days=days,
                                        beta=beta,
                                        average_dose=budget * .1 / n,
                                        draw=draw,
                                        policy=policy,
                                        infections=value,
                                        none=none,
                                        prevention_percent=100 * (none - value) / none))
            pd.DataFrame(records).to_csv(out / 'trials.csv', index=False)
            pd.DataFrame(calibration).to_csv(out / 'reference_calibration.csv', index=False)
            print('Moderate reference complete', days, beta, flush=True)
    summary = []
    for (days, beta), group in pd.DataFrame(records).groupby(['days', 'beta']):
        wide = group.pivot(index='draw', columns='policy', values='infections')
        none = group.none.iloc[0]
        for policy in wide:
            delta = wide.uninformed - wide[policy]
            half = t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
            summary.append(dict(days=days, beta=beta, policy=policy, infections=wide[policy].mean(), prevention_percent=100 * (none - wide[policy].mean()) / none,
                                extra_prevented=delta.mean(), percent_vs_uninformed=100 * delta.mean() / wide.uninformed.mean(), ci_low=delta.mean() - half, ci_high=delta.mean() + half))
    result = pd.DataFrame(summary)
    result.to_csv(out / 'summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(kind='Fully synthetic matched moderate-intervention benchmark', target='40% expected reduction under globally optimal static uninformed grid allocation',
                                                       calibration='Exact Uniform prior integrated by 1000-point quadrature; choose dose budget before any test response labels',
                                                       test='24 fresh Uniform(0,1) county-response maps, seeds1001000–1001023; same budget across information levels',
                                                       design='Full beta .24/.30/.36/.42 and horizons84/168, all retained; epidemic risk homogeneous; independent counties',
                                                       information='Known Uniform prior vs exact p vs correctly specified noisy measurements; not empirical survey evidence',
                                                       algorithm='Exact finite-grid static allocation dynamic programming; not adaptive PPO'), indent=2))
    print(result[result.policy.isin(['uninformed', 'perfect_information'])
                 ].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
