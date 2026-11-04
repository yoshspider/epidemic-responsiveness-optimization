"""Grid verification and observed-proxy sensitivity in the synthetic lab."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import ideal_information_benchmark as lab


def main():
    lab.DAYS = 84
    out = Path('outputs/ideal_information_benchmark_84d')
    # Independent discretization check at prespecified central budget.
    grid = []
    for step in [.1, .05]:
        doses = np.arange(0, 8 + step / 2, step)
        q = (np.arange(1000) + .5) / 1000
        curve = lab.outcomes(q[:, None], doses[None, :])
        units = round(32 / step)
        blind, _ = lab.optimize(np.tile(curve.mean(0), (32, 1)), units)
        for draw in range(24):
            p = np.random.default_rng(993000 + draw).uniform(0, 1, 32)
            cost = lab.outcomes(p[:, None], doses[None, :])
            oracle, minimum = lab.optimize(cost, units)
            baseline = cost[np.arange(32), blind].sum()
            assert minimum <= baseline + 1e-6
            grid.append(dict(step=step, draw=draw, baseline=baseline, oracle=minimum))
        print('Grid verification complete', step, flush=True)
    pd.DataFrame(grid).to_csv(out / 'grid_resolution_check.csv', index=False)
    # Preserve measured values: no artificial stretching of empirical p.
    cohort = pd.read_csv(
        'outputs/vaccine_experiment/eligible_counties.csv',
        dtype={
            'fips': str}).fips
    doses = np.arange(0, 8.05, .1)
    rows = []
    variation = []
    for metric in ['acceptance', 'recommendation']:
        series = {}
        for month in ['2021-02', '2021-03', '2021-04']:
            frame = pd.read_csv(
                f'data/raw/vaccine_experiment/{metric}_{month}.csv',
                dtype={
                    'geo_value': str})
            frame = frame[(frame.missing_value == 0) & (~frame.geo_value.str.endswith('000'))]
            series[month] = (
                frame.groupby('geo_value').value.mean() /
                100).reindex(cohort).to_numpy()
        for month, previous in [('2021-03', '2021-02'), ('2021-04', '2021-03')]:
            p = series[month]
            past = series[previous]
            count = len(p)
            if not np.isfinite(p).all() or not np.isfinite(past).all():
                raise ValueError('Missing proxy')
            cost = lab.outcomes(p[:, None], doses[None, :])
            past_cost = lab.outcomes(past[:, None], doses[None, :])
            # Shared current histogram: isolates knowledge of county assignment.
            ranked = np.empty(count)
            ranked[np.argsort(past, kind='stable')] = np.sort(p)
            rank_cost = lab.outcomes(ranked[:, None], doses[None, :])
            variation.append(
                dict(
                    metric=metric,
                    month=month,
                    counties=count,
                    mean=p.mean(),
                    sd=p.std(
                        ddof=1),
                    minimum=p.min(),
                    maximum=p.max()))
            for budget in [.25, .5, 1, 1.5, 2]:
                units = round(count * budget / .1)
                baseline_plan, baseline = lab.optimize(np.tile(cost.mean(0), (count, 1)), units)
                # Baseline cost is expectation over unknown random county labels,
                # not a conveniently chosen single ordering of those labels.
                oracle, oracle_value = lab.optimize(cost, units)
                raw, _ = lab.optimize(past_cost, units)
                rank_plan, _ = lab.optimize(rank_cost, units)
                none = float(lab.outcomes(np.array([.5]), np.array([0.]))[0] * count)
                for name, value in [('distribution_only', baseline), ('current_county_proxy', oracle_value),
                                    ('previous_month_raw', cost[np.arange(count), raw].sum()), ('previous_month_rank_with_shared_histogram', cost[np.arange(count), rank_plan].sum())]:
                    rows.append(dict(metric=metric, month=month, average_dose=budget, policy=name, infections=value,
                                     prevention_percent=100 * (none - value) / none, extra_prevented=baseline - value, improvement_percent=100 * (baseline - value) / baseline))
            print('Empirical proxy in synthetic lab complete', metric, month, flush=True)
    pd.DataFrame(rows).to_csv(out / 'empirical_proxy_sensitivity.csv', index=False)
    pd.DataFrame(variation).to_csv(out / 'empirical_proxy_variation.csv', index=False)
    (out / 'empirical_proxy_manifest.json').write_text(json.dumps(dict(kind='Observed survey values substituted as assumed responsiveness in synthetic equal-risk epidemic; NOT historical COVID replay or causal validation',
                                                                       proxies='Untransformed CTIS vaccine acceptance and recommendation receptiveness, February–April 2021, same 71 counties',
                                                                       epidemic='Each observed county assigned synthetic population 100000 and 10 initial infections; independent counties; 84 days',
                                                                       baseline='Globally optimal static expected allocation given exact current empirical histogram but no location labels; expectation over unknown label permutations',
                                                                       oracle='Globally optimal static grid allocation assuming current county survey fraction equals true responsiveness',
                                                                       raw_persistence='Previous-month observed county value, used directly; no fitted forecast model',
                                                                       rank_persistence='Previous-month county ranking calibrated to same current histogram known to baseline; isolates county ordering, NOT deployable forecasting claim',
                                                                       limitations='Survey attitudes are not measured causal intervention responsiveness; no real case forecasting or tweet predictor claim; maximum dose grid8'), indent=2))


if __name__ == '__main__':
    main()
