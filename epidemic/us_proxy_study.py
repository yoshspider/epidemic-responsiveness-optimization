"""Transport controlled response mechanism to U.S. geography, then replay.

All national results are conditional counterfactuals. Survey fractions are
assumed responsiveness surfaces, not measured causal effects of spending.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from .us_proxy_core import local_response_curves, interpolate_curves, cost_tables, horizon_allocation, simulate, checks


def load_surveys(counties):
    mask = pd.read_csv(
        'data/raw/nyt_mask_use.csv',
        dtype={
            'COUNTYFP': str}).set_index('COUNTYFP').reindex(
        counties.fips)
    hhs = pd.read_csv(
        'data/raw/national_hesitancy/county_hesitancy_compact.csv',
        dtype={
            'FIPS Code': str}).set_index('FIPS Code').reindex(
        counties.fips)
    surfaces = {'mask': (mask.ALWAYS + mask.FREQUENTLY + .5 * mask.SOMETIMES).to_numpy(),
                'vaccine': 1 - hhs['Estimated hesitant'].to_numpy()}
    for p in surfaces.values():
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid or missing national proxy')
    return surfaces


def rank_map(previous, target, n):
    order = np.argsort(previous, kind='stable')
    ranks = np.empty(len(n))
    ranks[order] = (np.cumsum(n[order]) - .5 * n[order]) / n.sum()
    order = np.argsort(target)
    cdf = (np.cumsum(n[order]) - .5 * n[order]) / n.sum()
    return np.interp(ranks, cdf, target[order])


def national_matrices(counties, out):
    n = len(counties)
    index = dict(zip(counties.fips, range(n)))
    matrices = []
    for week, date in enumerate(pd.date_range('2020-09-07', periods=12, freq='7D')):
        path = out / f'mobility_{week}.npz'
        if path.exists():
            matrix = sparse.load_npz(path)
        else:
            raw = pd.read_csv(
                f'data/raw/midterm_mobility/{date:%Y_%m_%d}.csv', dtype={'geoid_o': str, 'geoid_d': str})
            for col in ['geoid_o', 'geoid_d']:
                raw[col] = raw[col].str.zfill(5)
            raw = raw[raw.geoid_o.isin(index) & raw.geoid_d.isin(index)
                      & raw.geoid_o.ne(raw.geoid_d)]
            matrix = sparse.coo_matrix(
                (raw.pop_flows.to_numpy(float), (raw.geoid_d.map(index), raw.geoid_o.map(index))), shape=(
                    n, n)).tocsr()
            row = np.asarray(matrix.sum(1)).ravel()
            matrix = sparse.diags(np.divide(1, row, out=np.zeros_like(row), where=row > 0)) @ matrix
            sparse.save_npz(path, matrix)
        matrices.append(matrix)
    return matrices


def save_snapshots(out, name, result, dose):
    np.savez_compressed(out / f'{name}.npz', dose=dose, cumulative=np.asarray([w['cumulative'] for w in result['weeks']]),
                        active=np.asarray([w['active'] for w in result['weeks']]), daily_incidence=np.asarray([w['daily_incidence'] for w in result['weeks']]), daily_total=result['daily'])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical-only', action='store_true')
    args = parser.parse_args()
    checks()
    root = Path('outputs/us_proxy_study')
    root.mkdir(exist_ok=True)
    if args.historical_only:
        historical(root)
        return
    out = root / 'geographic'
    out.mkdir(exist_ok=True)
    counties = pd.read_csv(
        'data/transformed/census_population_2019.csv',
        dtype={
            'fips': str}).sort_values('fips').reset_index(
        drop=True)
    n = counties.population.to_numpy(float)
    s = n * .9999
    i = n * .0001
    r = np.zeros(len(n))
    counties.to_csv(out / 'counties.csv', index=False)
    np.savez_compressed(out / 'initial.npz', n=n, s=s, i=i, r=r)
    surveys = load_surveys(counties)
    matrices = national_matrices(counties, out)
    pd.DataFrame(
        dict(
            fips=counties.fips,
            mask=surveys['mask'],
            vaccine=surveys['vaccine'])).to_csv(
        out /
        'proxy_values.csv',
        index=False)
    agreement = dict(
        counties=len(n), pearson=float(
            np.corrcoef(
                surveys['mask'], surveys['vaccine'])[
                0, 1]), spearman=float(
                    spearmanr(
                        surveys['mask'], surveys['vaccine']).statistic))
    (out / 'cross_survey_agreement.json').write_text(json.dumps(agreement, indent=2))
    doses = np.arange(0, 8.0001, .1)
    zgrid = np.arange(0, 8.0001, .025)
    records = []
    prediction = pd.read_csv(
        'outputs/nyt_alternative/state_heldout_predictions.csv',
        dtype={
            'fips': str}).set_index('fips').reindex(
        counties.fips)
    for beta in [.24, .30, .36]:
        rates = np.full(12, beta)
        curves = local_response_curves(n, s, i, rates, zgrid, 84)
        for metric, p in surveys.items():
            known, prior = cost_tables(curves, zgrid, doses, p, n)
            forecasts = {
                'past_mask_raw': surveys['mask'], 'past_mask_rank': rank_map(
                    surveys['mask'], p, n)} if metric == 'vaccine' else {
                'demographic_prediction': prediction.forest_demographics.to_numpy()}
            forecast_costs = {name: interpolate_curves(curves, zgrid, np.clip(
                values, 0, 1)[:, None] * doses[None, :]) for name, values in forecasts.items()}
            for average in [.5, 1., 1.5]:
                choices = {'none': np.zeros(len(n)), 'uniform': np.full(len(n), average),
                           'distribution': horizon_allocation(prior, n, doses, average),
                           'proxy': horizon_allocation(known, n, doses, average)}
                choices.update({name: horizon_allocation(cost, n, doses, average)
                               for name, cost in forecast_costs.items()})
                for mobility in [.05, .15, .30]:
                    for policy, dose in choices.items():
                        result = simulate(n, s, i, r, p, dose, matrices, rates, mobility)
                        records.append(dict(metric=metric, beta=beta, average_dose=average, mobility=mobility, policy=policy,
                                            infections=result['total'], peak_daily=float(result['daily'].max()), peak_day=int(result['daily'].argmax() + 1), peak_at_end=bool(result['daily'].argmax() == 83)))
                        if beta == .30 and average == 1 and mobility == .15:
                            save_snapshots(out, f'{metric}_{policy}', result, dose)
                    pd.DataFrame(records).to_csv(out / 'trials.csv', index=False)
            print('National geographic grid complete', beta, metric, flush=True)
    table = pd.DataFrame(records)
    summaries = []
    for keys, group in table.groupby(['metric', 'beta', 'average_dose', 'mobility']):
        baseline = group.set_index('policy').loc['distribution', 'infections']
        none = group.set_index('policy').loc['none', 'infections']
        for row in group.to_dict('records'):
            row.update(extra_prevented=baseline - row['infections'], improvement_percent=100 * (baseline - row['infections']) / baseline,
                       prevented_vs_none=none - row['infections'], prevention_percent=100 * (none - row['infections']) / none)
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(kind='U.S.-GEOGRAPHY SYNTHETIC EPIDEMIC; NOT a historical COVID forecast or measured treatment effect',
                                                       counties=len(n), population=int(n.sum()), initial='Uniform .0001 prevalence, no prior immunity; synthetic expected states on actual Census2019 populations',
                                                       geography='Twelve actual weekly GeoDS networks September–November2020 used as a common transport structure, NOT an asserted calendar match to 2021 HHS attitudes',
                                                       intervention='Sustained static county dose, exp(-p*dose); same population-weighted budget for all nonzero policies; dose units not calibrated dollars',
                                                       proxies={
                                                           'mask': 'NYT July2020 ALWAYS+FREQUENTLY+.5*SOMETIMES',
                                                           'vaccine': 'HHS May26–June7 2021 non-hesitancy =1-Estimated hesitant; modeled county estimates, not observed causal responsiveness'},
                                                       comparison='Distribution baseline knows population-weighted full proxy histogram but not county labels; proxy-informed knows exact assigned surface',
                                                       forecasts='HHS: prior NYT mask values raw and rank-calibrated to the same aggregate histogram given baseline; masking: existing five-state-fold CDC SVI2018 forest reconstruction',
                                                       design='Full prespecified beta .24/.30/.36, average dose .5/1/1.5, mobility share .05/.15/.30; central .30,1,.15; no winner-only reporting',
                                                       allocator='Population-weighted lower-convex-envelope finite-grid horizon allocator, optimizing independent-county incidence then evaluating all policies in coupled network; not exact coupled optimum or PPO',
                                                       central='Snapshots and full 84-day incidence saved for all central policies',
                                                       uncertainty='Deterministic expected SIR; parameter grid is sensitivity, not a causal confidence interval'), indent=2))
    historical(root)
    print(
        pd.DataFrame(summaries).query('beta==.30 and average_dose==1 and mobility==.15').to_string(
            index=False),
        flush=True)


def historical(root):
    # Historical replay: only the preceding July2020 mask source is used.
    doses = np.arange(0, 8.0001, .1)
    zgrid = np.arange(0, 8.0001, .025)
    real = root / 'real_fall'
    initial = np.load(real / 'initial.npz')
    counties = pd.read_csv(real / 'counties.csv', dtype={'fips': str})
    n = initial['n']
    s = initial['s']
    i = initial['i']
    r = initial['r']
    p = load_surveys(counties)['mask']
    matrices = [sparse.load_npz(real / f'mobility_{w}.npz') for w in range(12)]
    rates = pd.read_csv(real / 'rates.csv').beta_total.to_numpy()
    curves = local_response_curves(n, s, i, rates, zgrid, 84)
    known, prior = cost_tables(curves, zgrid, doses, p, n)
    prediction = pd.read_csv(
        'outputs/nyt_alternative/state_heldout_predictions.csv',
        dtype={
            'fips': str}).set_index('fips').reindex(
        counties.fips).forest_demographics.to_numpy()
    predicted = interpolate_curves(curves, zgrid, prediction[:, None] * doses[None, :])
    records = []
    for average in [.25, .5, 1, 1.5]:
        for policy, dose in {'none': np.zeros(len(n)), 'uniform': np.full(len(n), average), 'distribution': horizon_allocation(prior, n, doses, average),
                             'proxy': horizon_allocation(known, n, doses, average), 'demographic_prediction': horizon_allocation(predicted, n, doses, average)}.items():
            result = simulate(n, s, i, r, p, dose, matrices, rates, .15)
            records.append(
                dict(
                    average_dose=average,
                    policy=policy,
                    infections=result['total'],
                    peak_day=int(
                        result['daily'].argmax() +
                        1)))
            if average == 1:
                save_snapshots(real, policy, result, dose)
    table = pd.DataFrame(records)
    rows = []
    for average, group in table.groupby('average_dose'):
        wide = group.set_index('policy')
        baseline = wide.loc['distribution', 'infections']
        none = wide.loc['none', 'infections']
        for row in group.to_dict('records'):
            row.update(extra_prevented=baseline - row['infections'], improvement_percent=100 * (baseline - row['infections']) / baseline,
                       prevented_vs_none=none - row['infections'], prevention_percent=100 * (none - row['infections']) / none)
            rows.append(row)
    pd.DataFrame(rows).to_csv(real / 'summary.csv', index=False)
    print('Historical replay complete', flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
