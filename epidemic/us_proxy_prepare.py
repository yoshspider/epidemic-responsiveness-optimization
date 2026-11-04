"""Retrospective September–November 2020 epidemic replay inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import least_squares


def main():
    out = Path('outputs/us_proxy_study/real_fall')
    out.mkdir(parents=True, exist_ok=True)
    population = pd.read_csv(
        'data/transformed/census_population_2019.csv',
        dtype={
            'fips': str}).sort_values('fips').reset_index(
        drop=True)
    reports = pd.read_csv('data/transformed/nyt_daily_reports.csv',
                          dtype={'fips': str}, parse_dates=['date'])
    known = population.fips.isin(reports.fips.unique()).to_numpy()
    counts = reports.pivot(
        index='date', columns='fips', values='cases').reindex(
        columns=population.fips).sort_index().ffill().fillna(0)
    n = population.population.to_numpy(float)
    infected = np.rint((counts.loc['2020-09-06'] - counts.loc['2020-08-30']).to_numpy() / .2)
    removed = np.rint(counts.loc['2020-08-30'].to_numpy() / .2)
    valid = known & np.isfinite(infected) & np.isfinite(removed) & (
        infected >= 0) & (
        removed >= 0) & (
            infected + removed <= n)
    population[~valid].to_csv(out / 'excluded_counties.csv', index=False)
    population = population[valid].reset_index(drop=True)
    n = n[valid]
    infected = infected[valid]
    removed = removed[valid]
    susceptible = n - infected - removed
    population.to_csv(out / 'counties.csv', index=False)
    np.savez_compressed(out / 'initial.npz', n=n, s=susceptible, i=infected, r=removed)
    index = dict(zip(population.fips, range(len(n))))
    matrices = []
    for week, date in enumerate(pd.date_range('2020-09-07', periods=12, freq='7D')):
        path = out / f'mobility_{week}.npz'
        matrix = sparse.load_npz(path) if path.exists() else None
        if matrix is None or matrix.shape != (len(n), len(n)):
            raw = pd.read_csv(
                f'data/raw/midterm_mobility/{date:%Y_%m_%d}.csv', dtype={'geoid_o': str, 'geoid_d': str})
            for col in ['geoid_o', 'geoid_d']:
                raw[col] = raw[col].str.zfill(5)
            raw = raw[raw.geoid_o.isin(index) & raw.geoid_d.isin(index)
                      & raw.geoid_o.ne(raw.geoid_d)]
            matrix = sparse.coo_matrix(
                (raw.pop_flows.to_numpy(float), (raw.geoid_d.map(index), raw.geoid_o.map(index))), shape=(
                    len(n), len(n))).tocsr()
            row = np.asarray(matrix.sum(1)).ravel()
            matrix = sparse.diags(np.divide(1, row, out=np.zeros_like(row), where=row > 0)) @ matrix
            sparse.save_npz(path, matrix)
        matrices.append(matrix)
    dates = pd.date_range('2020-09-07', periods=84)
    reports['target'] = reports.reported_incidence.where(
        reports.reported_incidence.ge(0) & (~reports.date_gap))
    data = reports.pivot(
        index='date',
        columns='fips',
        values='target').reindex(
        index=dates,
        columns=population.fips).to_numpy()
    s = susceptible.copy()
    i = infected.copy()
    r = removed.copy()
    rates = []
    series = []
    for week, matrix in enumerate(matrices):
        observed = data[7 * week:7 * week + 7]
        keep = np.isfinite(observed)

        def simulate(beta):
            ss = s.copy()
            ii = i.copy()
            rr = r.copy()
            values = []
            for day in range(7):
                h = beta * (.85 * ii / n + .15 * (matrix @ (ii / n)))
                new = ss * (-np.expm1(-h))
                recovered = ii * (-np.expm1(-1 / 7))
                ss -= new
                ii += new - recovered
                rr += recovered
                values.append(new * .2)
            return np.asarray(values), ss, ii, rr

        def residual(rate):
            estimate, *_ = simulate(rate[0])
            return np.sqrt(estimate[keep] + .375) - np.sqrt(observed[keep] + .375)
        fit = least_squares(residual, [.2], bounds=(0, 1.5), max_nfev=60)
        estimate, s, i, r = simulate(fit.x[0])
        rates.append(dict(week=week, beta_total=fit.x[0]))
        for day in range(7):
            series.append(dict(date=str(dates[7 * week + day].date()),
                               observed=np.nansum(observed[day]),
                               fitted=np.where(np.isfinite(observed[day]),
                                               estimate[day],
                                               0).sum()))
        print('Fall retrospective calibration', week + 1, round(fit.x[0], 4), flush=True)
    pd.DataFrame(rates).to_csv(out / 'rates.csv', index=False)
    pd.DataFrame(series).to_csv(out / 'calibration_series.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(kind='Retrospective empirical-input replay, NOT an outbreak forecast or causal effect estimation', dates=['2020-09-07', '2020-11-29'],
                                                       counties=len(n), population=int(n.sum()), reporting_fraction=.2, recovery='1/7 daily hazard', mobility_share=.15,
                                                       calibration='Contemporaneous weekly transmission fitted to observed reports; fixed 15% mobility split; all policies share same replay',
                                                       proxy='NYT July 2020 masking predates replay; treated as stable responsiveness trait, not a measured causal effect',
                                                       exclusions='Never separately reported/aggregated NYT geographies or invalid initial SIR; explicit excluded_counties.csv',
                                                       zero_case_initialization='Counties reported later in the source have zero cumulative cases before first report, with forward fill across reporting gaps; never-reported geographies remain excluded'), indent=2))


if __name__ == '__main__':
    main()
