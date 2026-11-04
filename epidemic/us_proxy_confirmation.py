"""Paired stochastic confirmation at fixed, prespecified central settings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
from .us_proxy_study import load_surveys
from .us_proxy_core import local_response_curves, interpolate_curves, cost_tables, horizon_allocation


def stochastic(n, s, i, r, p, dose, matrices, rates, seed):
    n = np.rint(n).astype(np.int64)
    i = np.rint(i).astype(np.int64)
    r = np.rint(r).astype(np.int64)
    s = n - i - r
    factor = np.exp(-p * dose)
    rho = -np.expm1(-1 / 7)
    total = 0
    daily = []
    for day in range(84):
        week = day // 7
        h = rates[week] * (.85 * i / n + .15 * (matrices[week] @ (i / n))) * factor
        # Separate reproducible streams by day and channel; pairing uses the
        # same initial state and seeds, not a claimed exact inverse-CDF coupling.
        new = np.random.default_rng(seed * 1000 + 2 * day).binomial(s, -np.expm1(-h))
        removed = np.random.default_rng(seed * 1000 + 2 * day + 1).binomial(i, rho)
        s -= new
        i += new - removed
        r += removed
        total += int(new.sum())
        daily.append(int(new.sum()))
    if not np.array_equal(s + i + r, n):
        raise ValueError('Integer mass conservation failed')
    return total, int(np.argmax(daily) + 1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['both', 'geographic', 'real_fall'], default='both')
    args = parser.parse_args()
    root = Path('outputs/us_proxy_study')
    doses = np.arange(0, 8.0001, .1)
    zgrid = np.arange(0, 8.0001, .025)
    rows = []
    if args.kind != 'both' and (root / 'confirmation_trials.csv').exists():
        previous = pd.read_csv(root / 'confirmation_trials.csv')
        rows = previous[previous.kind != args.kind].to_dict('records')
    for kind in (['geographic', 'real_fall'] if args.kind == 'both' else [args.kind]):
        folder = root / kind
        initial = np.load(folder / 'initial.npz')
        n = initial['n']
        s = initial['s']
        i = initial['i']
        r = initial['r']
        counties = pd.read_csv(folder / 'counties.csv', dtype={'fips': str})
        p = load_surveys(counties)['mask']
        matrices = [sparse.load_npz(folder / f'mobility_{w}.npz') for w in range(12)]
        rates = np.full(
            12, .3) if kind == 'geographic' else pd.read_csv(
            folder / 'rates.csv').beta_total.to_numpy()
        curves = local_response_curves(n, s, i, rates, zgrid, 84)
        known, prior = cost_tables(curves, zgrid, doses, p, n)
        prediction = pd.read_csv(
            'outputs/nyt_alternative/state_heldout_predictions.csv',
            dtype={
                'fips': str}).set_index('fips').reindex(
            counties.fips)
        tables = {'distribution': prior, 'proxy': known,
                  'demographic_prediction': interpolate_curves(curves, zgrid, prediction.forest_demographics.to_numpy()[:, None] * doses[None, :]),
                  'population_prediction': interpolate_curves(curves, zgrid, prediction.forest_population.to_numpy()[:, None] * doses[None, :])}
        for average in [.5, 1.]:
            allocations = {
                name: horizon_allocation(
                    cost,
                    n,
                    doses,
                    average) for name,
                cost in tables.items()}
            allocations['none'] = np.zeros(len(n))
            for replicate in range(8):
                seed = 996000 + replicate
                infected = np.minimum(np.random.default_rng(seed).poisson(
                    n * .0001), n).astype(float) if kind == 'geographic' else i
                susceptible = n - infected - r
                for policy, dose in allocations.items():
                    total, peak = stochastic(
                        n, susceptible, infected, r, p, dose, matrices, rates, seed)
                    rows.append(
                        dict(
                            kind=kind,
                            average_dose=average,
                            seed=replicate,
                            policy=policy,
                            infections=total,
                            peak_day=peak))
                pd.DataFrame(rows).to_csv(root / 'confirmation_trials.csv', index=False)
                print('Paired confirmation', kind, average, replicate + 1, flush=True)
    summary = []
    for (kind, average), group in pd.DataFrame(rows).groupby(['kind', 'average_dose']):
        wide = group.pivot(index='seed', columns='policy', values='infections')
        for policy in wide:
            for baseline in ['distribution', 'population_prediction']:
                delta = wide[baseline] - wide[policy]
                half = t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
                summary.append(dict(kind=kind, average_dose=average, policy=policy, baseline=baseline, infections=wide[policy].mean(),
                                    extra_prevented=delta.mean(), improvement_percent=100 * delta.mean() / wide[baseline].mean(), ci_low=delta.mean() - half, ci_high=delta.mean() + half,
                                    prevented_vs_none=(wide['none'] - wide[policy]).mean()))
    result = pd.DataFrame(summary)
    result.to_csv(root / 'confirmation_summary.csv', index=False)
    (root / 'confirmation_manifest.json').write_text(json.dumps(dict(kind='Conditional paired epidemic Monte Carlo, not causal confidence intervals',
                                                                     seeds='996000–996007, fixed before evaluating confirmation', settings='Masking proxy, budgets .5 and1; geography beta .3,mobility .15; calibrated historical replay',
                                                                     initialization='Geographic initial infections Poisson(.0001*N), shared across compared policies; historical fixed observed-data-derived SIR',
                                                                     policy='Static horizon plans fitted to expected initial scenario; no per-replicate policy retuning',
                                                                     pairing='Same initial state and day/channel RNG seeds; binomial generator consumption may differ across policies, not exact inverse-CDF common-uniform pairing',
                                                                     uncertainty='95% paired t intervals over eight epidemic seeds only; excludes survey measurement error, proxy validity, disease parameter and reporting uncertainty'), indent=2))
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
