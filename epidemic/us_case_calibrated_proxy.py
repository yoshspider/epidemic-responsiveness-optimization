"""Case-calibrated, stochastic proxy allocation benchmark for November 2020."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binom, t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .responsiveness_mechanisms import force, optimize

CALIBRATION = Path("outputs/weekly_case_calibration")
OUT = Path("outputs/case_calibrated_proxy")
DOSE = .8
REFERENCE = .5
DRAWS = 16


def predict(target, features, states):
    """Predict each state using labeled counties in other states only."""
    features = features.to_numpy()
    observed = np.isfinite(target)
    prediction = np.empty(len(target))
    for train, test in GroupKFold(5).split(features, groups=states):
        train = train[observed[train]]
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
        prediction[test] = model.fit(features[train], target[train]).predict(features[test])
    return np.clip(prediction, 0, 1)


def simulate(plan, truth, n, s, i, matrices, beta, m, uniforms):
    s, i = s.astype(np.int64), i.astype(np.int64)
    initial = s.sum()
    recovery = -np.expm1(-1 / 7)
    for day, (u_new, u_recovered) in enumerate(uniforms):
        week = day // 7
        h = force(i, n, truth, plan[week], matrices[week], beta, m, "recipient")
        new = binom.ppf(u_new, s, -np.expm1(-h)).astype(np.int64)
        recovered = binom.ppf(u_recovered, i, recovery).astype(np.int64)
        s -= new
        i += new - recovered
    return int(initial - s.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fitted = pd.read_csv(CALIBRATION / "summary.csv").set_index("model").loc["fit_both"]
    beta, m = fitted.beta_total, fitted.m
    with np.load(CALIBRATION / "initial_holdout.npz") as initial:
        n, s, i, fips = (initial[key] for key in ("n", "s", "i", "fips"))
    matrices = [sparse.load_npz(CALIBRATION / f"mobility_{w}.npz") for w in range(8, 12)]
    geography = pd.DataFrame({"fips": fips, "log_population": np.log(n)})
    mask = pd.read_csv("data/raw/nyt_mask_use.csv", dtype={"COUNTYFP": str}).set_index("COUNTYFP")
    july = (mask.ALWAYS + mask.FREQUENTLY + .5 * mask.SOMETIMES).reindex(fips).to_numpy()
    ctis = pd.read_csv("data/transformed/ctis_october2020_monthly_proxy.csv", dtype={"fips": str})
    ctis = ctis.loc[(ctis.month == "2020-10") & (ctis.observed_days >= 20)].set_index("fips")
    october = ctis.responsiveness.reindex(fips).to_numpy()
    tweets = pd.read_csv("outputs/tbcov_prediction/county_sentiment.csv", dtype={"fips": str})
    data = geography.merge(tweets, on="fips", validate="one_to_one", how="left")
    data["log_posts"] = np.log1p(data.post_count.fillna(0))
    states = data.fips.str[:2]
    observed = np.isfinite(october)
    if not np.isfinite(july).all():
        raise ValueError("NYT masking is missing for a calibration county")
    reference = np.full(len(n), REFERENCE)
    uniform = np.full((len(matrices), len(n)), DOSE)

    def plan(response, starts=()):
        return optimize(n, s, i, response, matrices, beta, m, "recipient",
                        DOSE, starts, 60, schedule="weekly")[0]

    baseline = plan(reference)
    scenarios = {
        "nyt_mask": (july, {"population_only": predict(july, data[["log_population"]], states)}),
        "ctis_mask": (np.where(observed, october, REFERENCE), {
            "population_only": predict(october, data[["log_population"]], states),
            "population_activity": predict(october, data[["log_population", "log_posts"]], states),
            "population_activity_sentiment": predict(
                october, data[["log_population", "log_posts", "mean_sentiment",
                               "positive_share", "negative_share"]], states),
        }),
    }
    records, prediction = [], []
    for scenario, (truth, predictors) in scenarios.items():
        plans = {"none": np.zeros_like(uniform), "uniform": uniform, "uninformed": baseline}
        for label, response in predictors.items():
            labeled = observed if scenario == "ctis_mask" else np.ones(len(n), dtype=bool)
            prediction.append(dict(scenario=scenario, policy=label, counties=int(labeled.sum()),
                                   rmse=np.sqrt(np.mean((response[labeled] - truth[labeled])**2))))
            # Keep the CTIS intervention experiment's missing-label convention
            # identical across all proxy policies and the oracle.
            planning = np.where(labeled, response, REFERENCE)
            plans[label] = plan(planning, [baseline])
        plans["surrogate_oracle"] = plan(truth, list(plans.values())[1:])
        print(f"{scenario}: allocation schedules optimized; running paired simulations", flush=True)
        for seed in range(1049200, 1049200 + DRAWS):
            rng = np.random.default_rng(seed)
            uniforms = rng.random((7*len(matrices), 2, len(n)))
            uniforms = np.maximum(uniforms, np.finfo(float).tiny)
            for label, schedule in plans.items():
                if label != "none":
                    np.testing.assert_allclose(schedule @ n, DOSE*n.sum(), rtol=1e-8)
                cases = simulate(schedule, truth, n, s, i, matrices, beta, m, uniforms)
                records.append(dict(scenario=scenario, seed=seed, policy=label, infections=cases))
        np.savez_compressed(OUT / f"{scenario}_plans.npz", fips=fips, **plans)

    trials = pd.DataFrame(records)
    summary = []
    for scenario, group in trials.groupby("scenario"):
        wide = group.pivot(index="seed", columns="policy", values="infections")
        reference_cases = wide.uninformed.mean()
        for label in wide.columns:
            differences = wide.uninformed - wide[label]
            half = t.ppf(.975, DRAWS - 1) * differences.std(ddof=1) / np.sqrt(DRAWS)
            summary.append(dict(scenario=scenario, policy=label, mean_infections=wide[label].mean(),
                                fewer_than_uninformed_pct=100*differences.mean()/reference_cases,
                                paired_extra_prevented=differences.mean(),
                                ci_low=differences.mean()-half, ci_high=differences.mean()+half))
    result = pd.DataFrame(summary)
    result.to_csv(OUT / "summary.csv", index=False)
    trials.to_csv(OUT / "trials.csv", index=False)
    pd.DataFrame(prediction).to_csv(OUT / "prediction.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        counties=len(n), ctis_counties=int(observed.sum()),
        ctis_tweet_counties=int((observed & data.mean_sentiment.notna()).sum()),
        beta=beta, m=m, weeks="2020-11-02 to 2020-11-29",
        uniform_dose=DOSE, uninformed_reference=REFERENCE, paired_draws=DRAWS,
        interpretation="Stochastic historical-input counterfactual with assumed masking response."
    ), indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
