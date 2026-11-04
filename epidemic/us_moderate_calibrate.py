"""Freeze moderate budgets using aggregate priors, never true county labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
from .us_proxy_study import load_surveys
from .us_proxy_core import simulate


def main():
    root = Path('outputs/us_proxy_study')
    out = root / 'moderate'
    out.mkdir(exist_ok=True)
    records = []
    for kind, metric, beta in [('geographic', 'mask', .36), ('geographic',
                                                             'vaccine', .36), ('real_fall', 'mask', .3)]:
        folder = root / kind
        initial = np.load(folder / 'initial.npz')
        n = initial['n']
        s = initial['s']
        i = initial['i']
        r = initial['r']
        counties = pd.read_csv(folder / 'counties.csv', dtype={'fips': str})
        p = load_surveys(counties)[metric]
        matrices = [sparse.load_npz(folder / f'mobility_{w}.npz') for w in range(12)]
        rates = np.full(
            12, beta) if kind == 'geographic' else pd.read_csv(
            folder / 'rates.csv').beta_total.to_numpy()
        order = np.argsort(p)
        cdf = np.cumsum(n[order]) / n.sum()
        worlds = []
        for draw in range(8):
            u = np.random.default_rng(1002000 + draw).uniform(0, 1, len(n))
            worlds.append(np.interp(u, cdf, p[order]))
        none = simulate(n, s, i, r, worlds[0], np.zeros(len(n)), matrices, rates, .15)['total']
        trace = []

        def objective(dose):
            value = np.mean([simulate(n, s, i, r, world, np.full(len(n), dose),
                            matrices, rates, .15)['total'] for world in worlds])
            prevention = 1 - value / none
            trace.append(dict(dose=dose, prevention_percent=100 * prevention))
            return prevention - .4
        chosen = float(brentq(objective, 0, 2, xtol=2e-5))
        records.append(
            dict(
                kind=kind,
                metric=metric,
                beta=beta,
                average_dose=chosen,
                target=40.,
                reference='Uniform per-capita intervention, expected over eight aggregate-prior maps'))
        pd.DataFrame(trace).to_csv(out / f'calibration_{kind}_{metric}.csv', index=False)
        pd.DataFrame(records).to_csv(out / 'budgets.csv', index=False)
        print('Moderate U.S. budget frozen', kind, metric, beta, chosen, flush=True)
    (out / 'manifest.json').write_text(json.dumps(dict(target='40% expected prevention for fixed population-proportional allocation under responsiveness uncertainty',
                                                       prior='Known population-weighted proxy histogram only; county labels randomized by fresh prior sampling, seeds1002000–1002007',
                                                       interpretation='Reference budget chosen without location-specific true responsiveness; optimized uninformed network policy is a stronger additional comparator and may exceed40%',
                                                       evaluation='All informed, predicted and uninformed policies receive exactly the frozen per-capita resource budget; actual county map is not used to tune dose'), indent=2))


if __name__ == '__main__':
    main()
