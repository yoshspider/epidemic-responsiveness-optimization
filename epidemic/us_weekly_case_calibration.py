"""Fit local and mobility transmission to NYT cases with a chronological holdout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .us_proxy_study import national_matrices


def main():
    out = Path("outputs/weekly_case_calibration")
    out.mkdir(parents=True, exist_ok=True)
    counties = pd.read_csv("data/transformed/census_population_2019.csv", dtype={"fips": str})
    reports = pd.read_csv("data/transformed/nyt_daily_reports.csv",
                          dtype={"fips": str}, parse_dates=["date"])
    counties = counties.loc[counties.fips.isin(reports.fips)].sort_values("fips").reset_index(drop=True)
    counts = reports.pivot(index="date", columns="fips", values="cases").sort_index()
    counts = counts.reindex(columns=counties.fips).ffill().fillna(0)
    targets = reports.assign(target=reports.reported_incidence.where(
        reports.reported_incidence.ge(0) & ~reports.date_gap))
    targets = targets.pivot(index="date", columns="fips", values="target")
    n = counties.population.to_numpy(float)
    matrices = national_matrices(counties, out)
    periods = []
    corrections = []
    for start, matrix in zip(pd.date_range("2020-09-07", periods=12, freq="7D"), matrices):
        recent = counts.loc[start - pd.Timedelta(days=1)].to_numpy()
        removed = counts.loc[start - pd.Timedelta(days=8)].to_numpy()
        valid = (recent >= removed) & (recent <= n) & (removed >= 0)
        # Negative revisions are not infections. Bound invalid initial states
        # for mobility, then omit those county-weeks from the fit and scoring.
        cumulative = np.clip(recent, 0, n)
        infected = cumulative - np.clip(removed, 0, cumulative)
        susceptible = n - cumulative
        observed = targets.reindex(index=pd.date_range(start, periods=7),
                                   columns=counties.fips).to_numpy(copy=True)
        observed[:, ~valid] = np.nan
        corrections.append({"week": str(start.date()), "excluded_county_weeks": int((~valid).sum())})
        periods.append((start, susceptible, infected, matrix, observed))

    def forecast(period, local, mobility):
        _, s, i, matrix, _ = period
        s, i = s.copy(), i.copy()
        daily = []
        for _ in range(7):
            prevalence = i / n
            hazard = local * prevalence + mobility * (matrix @ prevalence)
            new = s * -np.expm1(-hazard)
            i += new - i * -np.expm1(-1 / 7)
            s -= new
            daily.append(new)
        return np.asarray(daily)

    def residual(parameters, split=None, subset=periods[:8]):
        local, mobility = parameters if split is None else (parameters[0]*(1-split),
                                                            parameters[0]*split)
        errors = []
        for period in subset:
            observed = period[-1]
            keep = np.isfinite(observed)
            estimated = forecast(period, local, mobility)[keep]
            errors.append(np.sqrt(estimated + .375) - np.sqrt(observed[keep] + .375))
        return np.concatenate(errors)

    rows = []
    for name, split in (("fit_both", None), ("fixed_15_percent", .15), ("local_only", 0.)):
        start = [.17, .03] if split is None else [.2]
        fit = least_squares(residual, start, bounds=(0, 1.5), args=(split,))
        if not fit.success:
            raise RuntimeError(fit.message)
        local, mobility = fit.x if split is None else (fit.x[0]*(1-split), fit.x[0]*split)
        heldout = residual(fit.x, split, periods[8:])
        rows.append(dict(model=name, beta_local=local, beta_mobility=mobility,
                         beta_total=local+mobility, m=mobility/(local+mobility),
                         train_rmse=np.sqrt(np.mean(fit.fun**2)),
                         holdout_rmse=np.sqrt(np.mean(heldout**2))))
    result = pd.DataFrame(rows)
    result.to_csv(out / "summary.csv", index=False)
    counties.to_csv(out / "counties.csv", index=False)
    for name, week in (("initial_replay", 0), ("initial_holdout", 8)):
        _, susceptible, infected, _, _ = periods[week]
        np.savez_compressed(out / f"{name}.npz", n=n, s=susceptible, i=infected,
                            fips=counties.fips.to_numpy(dtype="U5"))
    (out / "manifest.json").write_text(json.dumps(dict(
        counties=len(n), train_weeks=[str(p[0].date()) for p in periods[:8]],
        holdout_weeks=[str(p[0].date()) for p in periods[8:]],
        initialization="Seven days of NYT reports are infectious; earlier reports removed; remaining residents susceptible. No multiplier.",
        evaluation="One-week predictions reinitialized with prior observed reports; no fitting to held-out outcomes.",
        loss="County-day RMSE on sqrt(reported cases + 0.375); missing and negative observations excluded.",
        revisions=corrections,
    ), indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
