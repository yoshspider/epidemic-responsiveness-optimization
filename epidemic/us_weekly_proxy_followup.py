"""Follow-up: evaluate October CTIS and percentile calibration on the same target."""

import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .responsiveness_mechanisms import objective, optimize
from .us_weekly_proxy import AVERAGE, BENCHMARK, OUT, SOURCE, percentile
from .us_proxy_core import cost_tables, horizon_allocation, local_response_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctis", default="data/transformed/ctis_october2020_monthly_proxy.csv")
    parser.add_argument("--acceptance", help="February 2021 CTIS vaccine-acceptance daily CSV")
    args = parser.parse_args()
    counties = pd.read_csv(SOURCE / "counties.csv", dtype={"fips": str})
    survey = pd.read_csv(SOURCE / "proxy_values.csv", dtype={"fips": str})
    ctis = pd.read_csv(args.ctis, dtype={"fips": str})
    ctis = ctis.loc[(ctis.month == "2020-10") & (ctis.observed_days >= 7),
                    ["fips", "responsiveness"]].rename(columns={"responsiveness": "ctis"})
    data = (counties.merge(survey, on="fips", validate="one_to_one", how="left")
            .merge(ctis, on="fips", validate="one_to_one", how="left"))
    if args.acceptance:
        daily = pd.read_csv(args.acceptance, dtype={"geo_value": str})
        keep = daily.time_value.between(20210201, 20210228) & (daily.issue < 20210526)
        daily = (daily.loc[keep].sort_values("issue")
                 .drop_duplicates(["geo_value", "time_value"], keep="last"))
        acceptance = daily.groupby("geo_value").value.agg(["mean", "count"])
        data["acceptance"] = data.fips.map(acceptance["mean"].where(acceptance["count"] >= 7)) / 100
    with np.load(SOURCE / "initial.npz") as initial:
        n, s, i = (initial[k] for k in ("n", "s", "i"))
    np.testing.assert_array_equal(data.population, n)
    matrices = [sparse.load_npz(SOURCE / f"mobility_{w}.npz") for w in range(12)]
    physics = (matrices, .36, .15, "recipient")
    baseline = np.load(BENCHMARK / "recipient_0.36_0.8_0.15_1011000_stratified_weekly"
                       / "selected_baseline.npz")["dose"]
    y = percentile(data.vaccine)
    groups = data.fips.str[:2]
    data["log_population"] = np.log(n)
    dose_grid, effect_grid = np.linspace(0, 8, 81), np.linspace(0, 8, 321)
    curves = local_response_curves(n, s, i, np.full(12, .36), effect_grid, 84)
    base_cases = objective(baseline, n, s, i, y, *physics, False)
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    feature_sets = {
        "population_rank": ["log_population"],
        "mask_population_rank": ["mask", "log_population"],
        "mask_population_ctis_rank": ["mask", "log_population", "ctis"],
    }
    if args.acceptance:
        feature_sets = {
            "mask_ctis_population_mean": ["mask", "ctis", "log_population"],
            "acceptance_population_mean": ["acceptance", "log_population"],
            "acceptance_mask_ctis_population_mean": ["acceptance", "mask", "ctis", "log_population"],
        }
    for name, features in feature_sets.items():
        x = data[features].to_numpy()
        predicted = np.empty(len(y))
        for train, test in GroupKFold(5).split(x, y, groups):
            model = HistGradientBoostingRegressor(
                max_iter=80, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1)
            predicted[test] = model.fit(x[train], y[train]).predict(x[test])
        # Use only the ordering of out-of-fold scores and the common Uniform prior.
        score = predicted if args.acceptance else percentile(predicted)
        known, _ = cost_tables(curves, effect_grid, dose_grid, score, n)
        local = horizon_allocation(known, n, dose_grid, AVERAGE)
        static, _ = optimize(n, s, i, score, *physics, AVERAGE,
                             [baseline.mean(axis=0), local], 80)
        plan, _ = optimize(n, s, i, score, *physics, AVERAGE,
                           [baseline, local, static], 80, schedule="weekly")
        value = objective(plan, n, s, i, y, *physics, False)
        rows.append(dict(policy=name, infections=value,
                         reduction_vs_uninformed_pct=100 * (base_cases - value) / base_cases,
                         spearman=spearmanr(score, y).statistic, rmse=np.sqrt(np.mean((score - y)**2)),
                         ctis_mask_counties=int(data.ctis.notna().sum()),
                         ctis_acceptance_counties=(int(data.acceptance.notna().sum())
                                                   if args.acceptance else None)))
        table = "acceptance_followup.csv" if args.acceptance else "calibration_followup.csv"
        pd.DataFrame(rows).to_csv(OUT / table, index=False)
        np.savez_compressed(OUT / f"{name}.npz", dose=plan, predicted=score)
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
