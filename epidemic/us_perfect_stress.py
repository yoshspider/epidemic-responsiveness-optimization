"""One-parameter stress tests of frozen plans, with no outcome-based retuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from scipy.stats import t
from .responsiveness_mechanisms import allocate, objective


def main():
    root = Path('outputs/us_perfect_information')
    out = root / 'confirmation'
    initial = np.load('outputs/us_proxy_study/geographic/initial.npz')
    n, s, i = initial['n'], initial['s'], initial['i']
    matrices = [
        sparse.load_npz(
            f'outputs/us_proxy_study/geographic/mobility_{week}.npz') for week in range(12)]
    cases = [('central', .36, .15, .8), ('budget_low', .36, .15, .6), ('budget_high', .36, .15, 1.),
             ('growth_low', .30, .15, .8), ('growth_high', .42, .15, .8),
             ('mobility_low', .36, .05, .8), ('mobility_high', .36, .30, .8)]
    rows = []
    for label, beta, mobility, budget in cases:
        none = objective(np.zeros(len(n)), n, s, i, np.full(len(n), .5),
                         matrices, beta, mobility, 'recipient', False)
        for seed in [1011000, 1011004, 1011008, 1011012]:
            folder = root / f'recipient_0.36_0.8_0.15_{seed}_stratified_weekly'
            baseline = np.load(folder / 'selected_baseline.npz')['dose']
            for draw in range(4):
                data = np.load(folder / f'oracle_{draw}.npz')
                p = data['p']
                oracle = data['dose']
                for policy, plan in [('uninformed', baseline), ('perfect', oracle)]:
                    if budget != .8:
                        plan = allocate(logit(np.clip(plan / 8, 1e-7, 1 - 1e-7)), n, budget)
                    value = objective(plan, n, s, i, p, matrices, beta,
                                      mobility, 'recipient', False)
                    rows.append(dict(case=label, beta=beta, mobility=mobility, budget=budget, world=seed + draw,
                                     policy=policy, infections=value, prevention_percent=100 * (none - value) / none))
        pd.DataFrame(rows).to_csv(out / 'stress_trials.csv', index=False)
        print('Frozen-plan stress complete', label, flush=True)
    summaries = []
    for label, group in pd.DataFrame(rows).groupby('case'):
        wide = group.pivot(index='world', columns='policy', values='infections')
        delta = wide.uninformed - wide.perfect
        half = t.ppf(.975, 15) * delta.std(ddof=1) / 4
        summaries.append(dict(case=label, uninformed=wide.uninformed.mean(), perfect=wide.perfect.mean(),
                              extra_avoided=delta.mean(), relative_reduction_percent=100 * delta.mean() / wide.uninformed.mean(), ci_low=delta.mean() - half, ci_high=delta.mean() + half))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'stress_summary.csv', index=False)
    (out / 'stress_manifest.json').write_text(json.dumps(dict(design='Six one-at-a-time perturbations around the frozen primary configuration',
                                                              policy='Original held-out weekly plans; no new optimization or model selection on stress outcomes',
                                                              budgets='For mean dose .6 or1, apply common log-odds shift to each weekly plan to satisfy exact new budget and cap8; preserve priorities',
                                                              growth='Actual beta .30/.42 while plans remain those chosen for .36', mobility='Actual share .05/.30 while plans remain those chosen for .15',
                                                              uncertainty='Sixteen held-out synthetic responsiveness maps; not empirical or causal uncertainty'), indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
