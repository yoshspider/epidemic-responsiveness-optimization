"""Evaluate a prior masking survey under the confirmed U.S. weekly allocation model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .responsiveness_mechanisms import objective, optimize
from .us_proxy_core import cost_tables, horizon_allocation, local_response_curves

SOURCE = Path("outputs/us_proxy_study/geographic")
BENCHMARK = Path("outputs/us_perfect_information")
OUT = Path("outputs/us_weekly_proxy")
AVERAGE = .8


def percentile(values):
    return (rankdata(values) - .5) / len(values)


def forecasts(mask, population, target, states):
    """Train on other states only; model settings fixed for both feature sets."""
    pop = np.log(population)[:, None]
    features = {"population_only": pop, "mask_plus_population": np.column_stack([pop, mask])}
    predictions = {}
    for name, x in features.items():
        pred = np.empty_like(target)
        for train, test in GroupKFold(5).split(x, target, states):
            model = HistGradientBoostingRegressor(
                max_iter=80, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1
            )
            pred[test] = model.fit(x[train], target[train]).predict(x[test])
        predictions[name] = pred
    return predictions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counties = pd.read_csv(SOURCE / "counties.csv", dtype={"fips": str})
    data = counties.merge(pd.read_csv(SOURCE / "proxy_values.csv", dtype={"fips": str}),
                          on="fips", how="left", validate="one_to_one", sort=False)
    with np.load(SOURCE / "initial.npz") as initial:
        n, s, i = (initial[key] for key in ("n", "s", "i"))
    np.testing.assert_array_equal(data.population, n)
    mask, vaccine = (data[key].to_numpy() for key in ("mask", "vaccine"))
    assert np.isfinite(mask).all() and np.isfinite(vaccine).all()
    matrices = [sparse.load_npz(SOURCE / f"mobility_{week}.npz") for week in range(12)]
    physics = (matrices, .36, .15, "recipient")
    frozen = np.load(BENCHMARK / "recipient_0.36_0.8_0.15_1011000_stratified_weekly"
                     / "selected_baseline.npz")["dose"]
    dose_grid = np.linspace(0, 8, 81)
    effect_grid = np.linspace(0, 8, 321)
    curves = local_response_curves(n, s, i, np.full(12, .36), effect_grid, 84)
    records, null, diagnostics = [], [], []
    for scale, truth, past in (
        ("percentile", percentile(vaccine), percentile(mask)), ("raw", vaccine, mask)
    ):
        baseline = frozen
        if scale == "raw":
            # Reveal the aggregate response histogram, without county labels.
            rng = np.random.default_rng(1049000)
            u = (np.argsort(rng.random((len(n), 8)), axis=1) + .5) / 8
            prior = np.quantile(truth, u)
            baseline, _ = optimize(n, s, i, prior, *physics, AVERAGE,
                                   [frozen], 80, schedule="weekly")

        def simulate(plan, response=truth):
            return objective(plan, n, s, i, response, *physics, False)

        def plan_for(response):
            known, _ = cost_tables(curves, effect_grid, dose_grid, response, n)
            local = horizon_allocation(known, n, dose_grid, AVERAGE)
            static, _ = optimize(n, s, i, response, *physics, AVERAGE,
                                 [baseline.mean(axis=0), local], 80)
            return optimize(n, s, i, response, *physics, AVERAGE,
                            [baseline, local, static], 80, schedule="weekly")[0]

        pred = forecasts(mask, n, truth, data.fips.str[:2])
        pred["mask_only"] = past
        pred["surrogate_oracle"] = truth
        plans = {"none": np.zeros_like(baseline), "uninformed": baseline}
        print(f"{scale}: optimizing schedules", flush=True)
        for name, response in pred.items():
            plans[name] = plan_for(response)
            diagnostics.append({"scale": scale, "policy": name,
                                "spearman": spearmanr(response, truth).statistic,
                                "rmse": np.sqrt(np.mean((response - truth)**2))})
            print(f"{scale}: {name} {simulate(plans[name]):,.0f}", flush=True)

        cases = {name: simulate(plan) for name, plan in plans.items()}
        if cases["surrogate_oracle"] > min(cases.values()):
            # The oracle can always adopt any feasible schedule from another policy.
            best = min(cases, key=cases.get)
            plans["surrogate_oracle"] = plans[best].copy()
            cases["surrogate_oracle"] = cases[best]
        gap = cases["uninformed"] - cases["surrogate_oracle"]
        for name, value in cases.items():
            gain = cases["uninformed"] - value
            records.append({"scale": scale, "policy": name, "infections": value,
                            "reduction_vs_none_pct": 100 * (1 - value / cases["none"]),
                            "reduction_vs_uninformed_pct": 100 * gain / cases["uninformed"],
                            "oracle_gap_recovered_pct": 100 * gain / gap if gap else np.nan})
        if scale == "percentile":
            # Independence falsification: retain the original, unrelated p maps.
            for seed in range(1011000, 1011016):
                p = np.random.default_rng(seed).uniform(0, 1, len(n))
                c0 = simulate(baseline, p)
                for name in ("mask_only", "mask_plus_population", "population_only"):
                    null.append({"seed": seed, "policy": name, "uninformed": c0,
                                 "infections": simulate(plans[name], p)})
        np.savez_compressed(OUT / f"{scale}_schedules.npz", fips=data.fips.to_numpy(dtype="U5"),
                            surrogate=truth, **plans)
        pd.DataFrame(records).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(OUT / "prediction.csv", index=False)
        pd.DataFrame(null).to_csv(OUT / "independent_worlds.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "assumption": "Vaccine non-hesitancy stands in for responsiveness; no causal labels.",
        "timing": "July 2020 masking predicts May–June 2021 attitudes. 2020 mobility is geometry only.",
        "population": int(n.sum()), "counties": len(n), "days": 84,
        "weekly_average_dose": AVERAGE, "beta": .36, "mobility_share": .15,
        "percentile": "Midrank transform gives survey rankings near-uniform scale.",
        "raw_prior": "Planner gets the aggregate raw target histogram, without county labels.",
        "validation": "Five groups of held-out states for the learned predictors; one observed target map."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
