"""Population-only and state-information controls over every geographic cell."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from .us_proxy_study import load_surveys
from .us_proxy_core import local_response_curves, interpolate_curves, horizon_allocation, simulate


def main():
    folder = Path('outputs/us_proxy_study/geographic')
    initial = np.load(folder / 'initial.npz')
    counties = pd.read_csv(folder / 'counties.csv', dtype={'fips': str})
    n = initial['n']
    s = initial['s']
    i = initial['i']
    r = initial['r']
    p = load_surveys(counties)['mask']
    matrices = [sparse.load_npz(folder / f'mobility_{week}.npz') for week in range(12)]
    frame = pd.read_csv(
        'outputs/nyt_alternative/state_heldout_predictions.csv',
        dtype={
            'fips': str}).set_index('fips').reindex(
        counties.fips)
    politics = pd.read_csv(
        'outputs/nyt_alternative_politics/state_heldout_predictions.csv',
        dtype={
            'fips': str}).set_index('fips').reindex(
        counties.fips)
    state = pd.DataFrame(dict(state=counties.fips.str[:2], p=p, n=n)).groupby(
        'state').apply(lambda g: np.average(g.p, weights=g.n))
    estimates = {'population_prediction': frame.forest_population.to_numpy(), 'state_mean': counties.fips.str[:2].map(state).to_numpy(),
                 'demographics_political': politics.forest_demographics_political.to_numpy()}
    doses = np.arange(0, 8.0001, .1)
    zgrid = np.arange(0, 8.0001, .025)
    rows = []
    for beta in [.24, .30, .36]:
        rates = np.full(12, beta)
        curves = local_response_curves(n, s, i, rates, zgrid, 84)
        for name, prediction in estimates.items():
            if not np.isfinite(prediction).all():
                raise ValueError('Missing held-out prediction')
            costs = interpolate_curves(curves, zgrid, prediction[:, None] * doses[None, :])
            for average in [.5, 1., 1.5]:
                dose = horizon_allocation(costs, n, doses, average)
                for mobility in [.05, .15, .30]:
                    value = simulate(n, s, i, r, p, dose, matrices, rates, mobility)['total']
                    rows.append(
                        dict(
                            beta=beta,
                            average_dose=average,
                            mobility=mobility,
                            policy=name,
                            infections=value))
            pd.DataFrame(rows).to_csv(folder / 'control_grid.csv', index=False)
        print('Full controls grid beta complete', beta, flush=True)
    original = pd.read_csv(folder / 'summary.csv')
    original = original[original.metric == 'mask']
    combined = pd.concat([original[['beta', 'average_dose', 'mobility', 'policy',
                         'infections']], pd.DataFrame(rows)], ignore_index=True)
    output = []
    for (beta, average, mobility), group in combined.groupby(['beta', 'average_dose', 'mobility']):
        base = group.set_index('policy').loc['population_prediction', 'infections']
        for row in group.to_dict('records'):
            row['gain_vs_population'] = base - row['infections']
            row['percent_vs_population'] = 100 * (base - row['infections']) / base
            output.append(row)
    pd.DataFrame(output).to_csv(folder / 'full_comparison_grid.csv', index=False)


if __name__ == '__main__':
    main()
