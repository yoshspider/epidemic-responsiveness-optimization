"""Separate outbreak growth, duration and responsiveness scaling in the US replay."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from .responsiveness_mechanisms import objective, optimize
from .us_case_calibrated_proxy import CALIBRATION, DOSE, REFERENCE, simulate, predict


def main():
    out = Path("outputs/case_calibration_bridge")
    out.mkdir(parents=True, exist_ok=True)
    fit = pd.read_csv(CALIBRATION / "summary.csv").set_index("model").loc["fit_both"]
    with np.load(CALIBRATION / "initial_holdout.npz") as initial:
        n, s, i, fips = (initial[key] for key in ("n", "s", "i", "fips"))
    flows = [sparse.load_npz(CALIBRATION / f"mobility_{w}.npz") for w in range(8, 12)]
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
    for beta in (fit.beta_total, .36):
        for blocks in (1, 3):
            matrices = flows * blocks
            shape = (len(matrices), len(n))
            args = (n, s, i)
            physics = (matrices, beta, fit.m, "recipient")
            zero = np.zeros(shape)

            def plan(p, starts=()):
                return optimize(*args, p, *physics, DOSE, starts, 60, schedule="weekly")[0]

            blind = plan(np.full(len(n), REFERENCE))
            for key, truth in responses.items():
                proxy = plan(predictions[key], [blind])
                oracle = plan(truth, [blind, proxy])
                policies = {"none": zero, "uninformed": blind, "population_proxy": proxy,
                            "oracle": oracle}
                if key == "raw":
                    # Diagnostic only: reveal the aggregate score mean, not county labels.
                    policies["mean_only_diagnostic"] = plan(np.full(len(n), truth.mean()), [blind])
                cases = {name: objective(dose, *args, truth, *physics, False)
                         for name, dose in policies.items()}
                for name, value in cases.items():
                    rows.append(dict(beta=beta, days=7*len(matrices), response=key,
                                     policy=name, infections=value,
                                     fewer_vs_uninformed_pct=100*(1-value/cases["uninformed"])))
                print(beta, 7*len(matrices), key, {p: round(c) for p, c in cases.items()}, flush=True)
                pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
                if beta == fit.beta_total and blocks == 1 and key == "raw":
                    paired = []
                    for seed in range(1049200, 1049216):
                        uniforms = np.random.default_rng(seed).random((28, 2, len(n)))
                        for name, schedule in policies.items():
                            value = simulate(schedule, truth, n, s, i, matrices, beta, fit.m, uniforms)
                            paired.append(dict(seed=seed, policy=name, infections=value))
                    pd.DataFrame(paired).to_csv(out / "historical_paired.csv", index=False)


if __name__ == "__main__":
    main()
