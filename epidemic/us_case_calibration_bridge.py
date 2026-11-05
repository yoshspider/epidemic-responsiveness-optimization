"""Separate outbreak growth, duration and responsiveness scaling in the US replay."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from .responsiveness_mechanisms import objective, optimize
from .us_case_calibrated_proxy import CALIBRATION, DOSE, REFERENCE, simulate, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--long-horizon", action="store_true",
                       help="Evaluate 24 and 48 weeks using the case-fitted transmission rate")
    modes.add_argument("--growth-grid", action="store_true",
                       help="Evaluate raw masking over 12 weeks at three intermediate rates")
    modes.add_argument("--historical-budgets", action="store_true",
                       help="Compare weekly doses using the 12-week case-fitted replay")
    parser.add_argument("--continuous-calibration", action="store_true",
                        help="Use rates fitted without weekly resets in the historical budget sweep")
    settings = parser.parse_args()
    if settings.continuous_calibration and not settings.historical_budgets:
        parser.error("--continuous-calibration requires --historical-budgets")
    out = Path("outputs/case_calibration_bridge")
    if settings.long_horizon:
        out = out.with_name("case_calibration_long_horizon")
    elif settings.growth_grid:
        out = out.with_name("case_calibration_growth_grid")
    elif settings.historical_budgets:
        out = out.with_name("case_calibration_historical_budgets")
        if settings.continuous_calibration:
            out = out.with_name("case_calibration_continuous_budgets")
    out.mkdir(parents=True, exist_ok=True)
    calibration_file, calibration_model = ("continuous_summary.csv", "continuous_fit") if (
        settings.continuous_calibration) else ("summary.csv", "fit_both")
    fit = pd.read_csv(CALIBRATION / calibration_file).set_index("model").loc[calibration_model]
    initial_file = "initial_replay.npz" if settings.historical_budgets else "initial_holdout.npz"
    with np.load(CALIBRATION / initial_file) as initial:
        n, s, i, fips = (initial[key] for key in ("n", "s", "i", "fips"))
    first_week = 0 if settings.historical_budgets else 8
    flows = [sparse.load_npz(CALIBRATION / f"mobility_{w}.npz") for w in range(first_week, 12)]
    mask = pd.read_csv("data/raw/nyt_mask_use.csv", dtype={"COUNTYFP": str}).set_index("COUNTYFP")
    raw = (mask.ALWAYS + mask.FREQUENTLY + .5 * mask.SOMETIMES).reindex(fips).to_numpy()
    if not np.isfinite(raw).all():
        raise ValueError("Missing NYT masking score")
    rank = (rankdata(raw) - .5) / len(raw)
    responses = {"raw": raw, "rank": rank}
    features = pd.DataFrame({"log_population": np.log(n)})
    states = np.array([code[:2] for code in fips])
    predictions = {key: predict(value, features, states)
                   for key, value in responses.items()}
    rows = []
    rates = (fit.beta_total,) if settings.long_horizon else (fit.beta_total, .36)
    lengths = (6, 12) if settings.long_horizon else (1, 3)
    if settings.growth_grid:
        rates, lengths = (.22, .26, .30), (3,)
        responses = {"raw": raw}
    budgets = (DOSE,)
    if settings.historical_budgets:
        rates, lengths = (fit.beta_total,), (1,)
        responses = {"raw": raw}
        budgets = (.05, .15, .35, DOSE)
    for beta, blocks, budget in ((b, w, d) for b in rates for w in lengths for d in budgets):
        matrices = flows * blocks
        shape = (len(matrices), len(n))
        args = (n, s, i)
        physics = (matrices, beta, fit.m, "recipient")
        zero = np.zeros(shape)

        def plan(p, starts=()):
            return optimize(*args, p, *physics, budget, starts, 60, schedule="weekly")[0]

        blind = plan(np.full(len(n), REFERENCE))
        for key, truth in responses.items():
            proxy = plan(predictions[key], [blind])
            policies = {"none": zero, "uninformed": blind, "population_proxy": proxy}
            oracle_starts = [blind, proxy]
            if key == "raw":
                # Diagnostic only: reveal the aggregate score mean, not county labels.
                policies["mean_only_diagnostic"] = plan(np.full(len(n), truth.mean()), [blind])
                if settings.historical_budgets:
                    oracle_starts.append(policies["mean_only_diagnostic"])
            policies["oracle"] = plan(truth, oracle_starts)
            cases = {name: objective(dose, *args, truth, *physics, False)
                     for name, dose in policies.items()}
            for name, value in cases.items():
                row = dict(beta=beta, days=7*len(matrices), response=key, policy=name,
                           infections=value,
                           fewer_vs_uninformed_pct=100*(1-value/cases["uninformed"]))
                if settings.historical_budgets:
                    row["weekly_dose"] = budget
                rows.append(row)
            print(beta, 7*len(matrices), budget, key,
                  {p: round(c) for p, c in cases.items()}, flush=True)
            pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
            if not settings.historical_budgets and beta == fit.beta_total and blocks == 1 and key == "raw":
                paired = []
                for seed in range(1049200, 1049216):
                    uniforms = np.random.default_rng(seed).random((28, 2, len(n)))
                    for name, schedule in policies.items():
                        value = simulate(schedule, truth, n, s, i, matrices, beta, fit.m, uniforms)
                        paired.append(dict(seed=seed, policy=name, infections=value))
                pd.DataFrame(paired).to_csv(out / "historical_paired.csv", index=False)


if __name__ == "__main__":
    main()
