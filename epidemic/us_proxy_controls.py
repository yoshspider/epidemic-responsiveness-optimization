"""Stronger same-input controls and proxy reliability sensitivity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from .us_proxy_study import load_surveys, save_snapshots
from .us_proxy_core import local_response_curves, interpolate_curves, cost_tables, horizon_allocation, simulate


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical-only', action='store_true')
    args = parser.parse_args()
    root = Path('outputs/us_proxy_study')
    doses = np.arange(0, 8.0001, .1)
    zgrid = np.arange(0, 8.0001, .025)
    for kind in (['real_fall'] if args.historical_only else ['geographic', 'real_fall']):
        folder = root / kind
        initial = np.load(folder / 'initial.npz')
        counties = pd.read_csv(folder / 'counties.csv', dtype={'fips': str})
        n = initial['n']
        s = initial['s']
        i = initial['i']
        r = initial['r']
        p = load_surveys(counties)['mask']
        matrices = [sparse.load_npz(folder / f'mobility_{w}.npz') for w in range(12)]
        rates = np.full(
            12, .3) if kind == 'geographic' else pd.read_csv(
            folder / 'rates.csv').beta_total.to_numpy()
        curves = local_response_curves(n, s, i, rates, zgrid, 84)
        predictions = pd.read_csv(
            'outputs/nyt_alternative/state_heldout_predictions.csv',
            dtype={
                'fips': str}).set_index('fips').reindex(
            counties.fips)
        # National aggregate information is also supplied to all predictors.
        state_mean = pd.DataFrame(dict(state=counties.fips.str[:2], p=p, n=n)).groupby(
            'state').apply(lambda g: np.average(g.p, weights=g.n))
        controls = {'population_prediction': predictions.forest_population.to_numpy(),
                    'state_mean': counties.fips.str[:2].map(state_mean).to_numpy()}
        rows = []
        for name, estimate in controls.items():
            cost = interpolate_curves(curves, zgrid, estimate[:, None] * doses[None, :])
            for average in ([.5, 1, 1.5] if kind == 'geographic' else [.25, .5, 1, 1.5]):
                dose = horizon_allocation(cost, n, doses, average)
                result = simulate(n, s, i, r, p, dose, matrices, rates, .15)
                rows.append(
                    dict(
                        kind=kind,
                        average_dose=average,
                        policy=name,
                        infections=result['total']))
                if average == 1:
                    save_snapshots(
                        folder,
                        ('mask_' if kind == 'geographic' else '') +
                        name,
                        result,
                        dose)
        pd.DataFrame(rows).to_csv(folder / 'stronger_controls.csv', index=False)
        print('Stronger controls complete', kind, flush=True)
    if args.historical_only:
        return
    # Preserve marginal true response values while perturbing their county labels.
    folder = root / 'geographic'
    initial = np.load(folder / 'initial.npz')
    counties = pd.read_csv(folder / 'counties.csv', dtype={'fips': str})
    n = initial['n']
    s = initial['s']
    i = initial['i']
    r = initial['r']
    p0 = load_surveys(counties)['mask']
    matrices = [sparse.load_npz(folder / f'mobility_{w}.npz') for w in range(12)]
    rates = np.full(12, .3)
    curves = local_response_curves(n, s, i, rates, zgrid, 84)
    proxy_cost = interpolate_curves(curves, zgrid, p0[:, None] * doses[None, :])
    proxy_dose = horizon_allocation(proxy_cost, n, doses, 1.)
    rows = []
    for alignment in [0., .25, .5, .75, 1.]:
        for draw in range(8):
            rng = np.random.default_rng(995000 + draw)
            p = p0.copy()
            indexes = rng.permutation(len(n))[:round((1 - alignment) * len(n))]
            p[indexes] = p[indexes][rng.permutation(len(indexes))]
            _, prior = cost_tables(curves, zgrid, doses, p, n)
            baseline_dose = horizon_allocation(prior, n, doses, 1.)
            base = simulate(n, s, i, r, p, baseline_dose, matrices, rates, .15)['total']
            informed = simulate(n, s, i, r, p, proxy_dose, matrices, rates, .15)['total']
            rows.append(dict(alignment_fraction=alignment, draw=draw, baseline=base,
                        proxy=informed, gain=base - informed, percent=100 * (base - informed) / base))
        pd.DataFrame(rows).to_csv(folder / 'alignment_sensitivity.csv', index=False)
        print('Proxy alignment sensitivity complete', alignment, flush=True)
    (root / 'controls_manifest.json').write_text(json.dumps(dict(controls='State-held-out population-only mask predictor and state aggregate mask response surface; same horizon allocator and budget',
                                                                 interpretation='Controls evaluate whether demographics adds value beyond population already present in epidemic state, or beyond state-level behavior information',
                                                                 reliability='Randomly reassign proxy-derived true responsiveness among a subset of county labels, preserving unweighted response distribution; baseline receives true global population-weighted histogram',
                                                                 limitation='Alignment fractions are simulated stress tests, not estimated reliability of real survey responsiveness'), indent=2))


if __name__ == '__main__':
    main()
