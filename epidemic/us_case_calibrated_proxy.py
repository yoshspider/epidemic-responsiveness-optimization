"""Case-calibrated, stochastic proxy allocation benchmark for September–November 2020."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
from scipy.stats import binom, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .responsiveness_mechanisms import RECOVERY_PROBABILITY, force, objective, optimize

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
    weekly_beta = np.broadcast_to(np.asarray(beta, dtype=float), (len(matrices),))
    for day, (u_new, u_recovered) in enumerate(uniforms):
        week = day // 7
        h = force(i, n, truth, plan[week], matrices[week], weekly_beta[week], m, "recipient")
        new = binom.ppf(u_new, s, -np.expm1(-h)).astype(np.int64)
        recovered = binom.ppf(u_recovered, i, RECOVERY_PROBABILITY).astype(np.int64)
        s -= new
        i += new - recovered
    return int(initial - s.sum())


def main():
    """Evaluate three planners on one response surface and one historical window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-baseline", type=float,
                        help="Choose the budget to approach this epidemic-only infection total")
    settings = parser.parse_args()
    out = OUT.with_name("case_calibrated_main")
    if settings.target_baseline is not None:
        out = out.with_name(f"{out.name}_target_{settings.target_baseline:g}")
    out.mkdir(parents=True, exist_ok=True)
    fitted = pd.read_csv(CALIBRATION / "weekly_tracking_fit.csv")
    beta, mobility = fitted.beta.to_numpy(), float(fitted.mobility_share.iloc[0])
    with np.load(CALIBRATION / "initial_replay.npz") as initial:
        n, s, i, fips = (initial[key] for key in ("n", "s", "i", "fips"))
    matrices = [sparse.load_npz(CALIBRATION / f"mobility_{week}.npz") for week in range(12)]
    mask = pd.read_csv("data/raw/nyt_mask_use.csv", dtype={"COUNTYFP": str}).set_index("COUNTYFP")
    july = (mask.ALWAYS + mask.FREQUENTLY + .5 * mask.SOMETIMES).reindex(fips).to_numpy()
    if not np.isfinite(july).all():
        raise ValueError("NYT masking is missing for a calibration county")
    truth = (rankdata(july) - .5) / len(july)

    columns = ["E_PCI", "EP_POV", "EP_UNEMP", "EP_NOHSDP", "EP_AGE65", "EP_DISABL",
               "EP_SNGPNT", "EP_LIMENG", "EP_NOVEH", "EP_UNINSUR", "AREA_SQMI"]
    svi = pd.read_csv("data/raw/alternative_sources/cdc_svi2018_counties.csv", dtype={"FIPS": str})
    features = svi.set_index("FIPS")[columns].reindex(fips).apply(pd.to_numeric, errors="coerce")
    features = features.where(features >= 0)
    features["log_population"] = np.log(n)
    features["log_density"] = np.log1p(n / features.pop("AREA_SQMI").replace(0, np.nan))
    states = [code[:2] for code in fips]
    predicted = predict(truth, features, states)

    def plan(response, starts=(), *, dose=DOSE):
        return optimize(n, s, i, response, matrices, beta, mobility, "recipient",
                        dose, starts, 60, schedule="weekly")[0]

    reference = np.full(len(n), REFERENCE)
    dose = DOSE
    if settings.target_baseline is None:
        epidemic_only = plan(reference)
    else:
        target = settings.target_baseline
        if not np.isfinite(target) or target <= 0:
            parser.error("--target-baseline must be positive and finite")

        def baseline_gap(budget):
            schedule = (plan(reference, dose=budget) if budget else
                        np.zeros((len(matrices), len(n))))
            infections = objective(schedule, n, s, i, truth, matrices,
                                   beta, mobility, "recipient", gradient=False)
            print(f"Dose {budget:.5f}: epidemic-only mean-count total {infections:,.0f}",
                  flush=True)
            return infections - target

        dose = brentq(baseline_gap, 0, DOSE, xtol=.0005)
        epidemic_only = plan(reference, dose=dose)

    county_predictor = plan(predicted, [epidemic_only], dose=dose)
    plans = {
        "epidemic_only": epidemic_only,
        "county_predictor": county_predictor,
        "exact_county_scores": plan(truth, [epidemic_only, county_predictor], dose=dose),
    }
    for schedule in plans.values():
        np.testing.assert_allclose(schedule @ n, dose * n.sum(), rtol=1e-8)

    print("Three schedules optimized. Evaluating paired stochastic trials.", flush=True)
    records = []
    for seed in range(1049200, 1049200 + DRAWS):
        uniforms = np.random.default_rng(seed).random((7 * len(matrices), 2, len(n)))
        uniforms = np.maximum(uniforms, np.finfo(float).tiny)
        for label, schedule in plans.items():
            cases = simulate(schedule, truth, n, s, i, matrices, beta, mobility, uniforms)
            records.append(dict(seed=seed, policy=label, infections=cases))
    trials = pd.DataFrame(records)
    wide = trials.pivot(index="seed", columns="policy", values="infections")
    mean_infections = wide.loc[:, list(plans)].mean()
    result = mean_infections.rename_axis("policy").rename("mean_infections").reset_index()
    result["fewer_than_epidemic_only_pct"] = (
        100 * (1 - result.mean_infections / mean_infections["epidemic_only"])
    )
    result.to_csv(out / "summary.csv", index=False)
    trials.to_csv(out / "trials.csv", index=False)
    np.savez_compressed(out / "plans.npz", fips=fips, **plans)
    (out / "manifest.json").write_text(json.dumps(dict(
        counties=len(n), weekly_beta=beta.tolist(), mobility_share=mobility,
        weeks="2020-09-07 to 2020-11-29", calibration="Retrospective weekly national case tracking",
        calibration_overlap=True, weekly_dose=dose, target_baseline=settings.target_baseline,
        epidemic_only_reference=REFERENCE,
        paired_draws=DRAWS, target="NYT July masking percentiles as assumed response",
        features=features.columns.tolist(),
    ), indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
