import argparse
import json
import hashlib
from importlib.metadata import version, PackageNotFoundError
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .model import Model, Config
from .data import load_data, synthetic
from .policies import allocate


def run(model, initial, policy, days, seed, predicted=None, record=True):
    if not np.isclose(days / model.cfg.dt, round(days / model.cfg.dt)):
        raise ValueError("Duration must be divisible by dt")
    model.validate_state(initial)
    state = initial.copy()
    rng = np.random.default_rng(seed)
    steps = round(days / model.cfg.dt)
    interval = round(model.cfg.decision_days / model.cfg.dt)
    week_steps = round(7 / model.cfg.dt)
    remaining = model.cfg.budget if model.cfg.budget_mode == "lump" else 0.0
    logs, trajectory = [], []
    cumulative = np.zeros_like(model.n)
    spending = 0.0
    allocations = np.zeros(len(model.n))
    peak = float(state.i.sum())
    for step in range(steps):
        if model.cfg.budget_mode == "weekly" and step % week_steps == 0:
            remaining = model.cfg.budget  # Unspent weekly funds expire; no carryover.
        if step % interval == 0:
            action, decisions = allocate(model, state, remaining, policy, predicted, steps-step)
            cost = float(action @ model.cost)
            if cost > remaining + 1e-7:
                raise RuntimeError("Budget overspend")
            remaining = max(0.0, remaining-cost)
            spending += cost
            allocations += action
            state.effort += action
            for decision in decisions if record else []:
                logs.append({"seed": seed, "policy": policy, "day": step*model.cfg.dt, **decision})
        # Fixed draws per step, county, group and channel, independent of policy actions.
        uniforms = (rng.uniform(1e-12, 1-1e-12, model.n.shape),
                    rng.uniform(1e-12, 1-1e-12, model.n.shape),
                    rng.uniform(1e-12, 1-1e-12, len(model.n)))
        state, new = model.step(state, uniforms=uniforms if model.cfg.stochastic or model.cfg.stochastic_adoption else None)
        cumulative += new
        peak = max(peak, float(state.i.sum()))
        for j in range(len(model.n)) if record else []:
            trajectory.append({"seed": seed, "policy": policy, "day": (step+1)*model.cfg.dt,
                               "county_index": j, "new_general": new[j,0], "new_vulnerable": new[j,1],
                               "susceptible": state.s[j].sum(), "infected": state.i[j].sum(),
                               "recovered": state.r[j].sum(), "effort": state.effort[j]})
    model.validate_state(state)
    metrics = {"seed": seed, "policy": policy, "infections": float(cumulative.sum()),
               "vulnerable": float(cumulative[:,1].sum()), "deaths": float((cumulative*np.array(model.cfg.ifr)).sum()),
               "peak_infected": peak, "spent": spending}
    return metrics, logs, trajectory, allocations


def paired_summary(metrics):
    rows = []
    for outcome in ("infections", "vulnerable", "deaths"):
        wide = metrics.pivot(index="seed", columns="policy", values=outcome)
        for left, right in combinations(wide.columns, 2):
            delta = (wide[left]-wide[right]).to_numpy()
            mean = delta.mean()
            half = t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
            rows.append({"outcome": outcome, "left": left, "right": right,
                         "mean_left_minus_right": mean, "ci_low": mean-half, "ci_high": mean+half,
                         "seeds": len(delta)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--counties", type=int, default=24)
    parser.add_argument("--days", type=float, default=84)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--policies", nargs="+", default=["none", "epidemic", "oracle", "lookahead"])
    args = parser.parse_args()
    if bool(args.data) == args.synthetic or args.seeds < 1 or args.days <= 0:
        parser.error("Choose exactly one of --data or --synthetic; positive duration and seeds required")
    cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
    frame, n, p, cost, mobility, initial, predicted = synthetic(args.counties) if args.synthetic else load_data(args.data, cfg.start_date)
    model = Model(n, p, cost, mobility, cfg)
    if "none" not in args.policies:
        args.policies.insert(0, "none")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    metrics, overlaps = [], []
    written = set()
    for seed in range(args.seeds):
        actions = {}
        for policy in args.policies:
            print(f"seed={seed} policy={policy}", flush=True)
            m, l, tr, a = run(model, initial, policy, args.days, seed, predicted)
            metrics.append(m)
            for name, rows in (("allocations", l), ("trajectory", tr)):
                if rows:
                    table = pd.DataFrame(rows)
                    table["fips"] = table.county_index.map(dict(enumerate(frame.fips)))
                    table.to_csv(out / f"{name}.csv", index=False,
                                 mode="a" if name in written else "w", header=name not in written)
                    written.add(name)
            actions[policy] = a
        for left, right in combinations(actions, 2):
            a, b = actions[left], actions[right]
            union = np.count_nonzero((a>0)|(b>0))
            overlaps.append({"seed": seed, "left": left, "right": right,
                             "jaccard": np.count_nonzero((a>0)&(b>0))/union if union else 1.,
                             "effort_l1": float(abs(a-b).sum())})
    metrics = pd.DataFrame(metrics)
    baseline = metrics[metrics.policy == "none"].set_index("seed").infections
    metrics["infections_prevented"] = metrics.seed.map(baseline)-metrics.infections
    metrics.to_csv(out / "metrics.csv", index=False)
    paired_summary(metrics).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(overlaps).to_csv(out / "allocation_overlap.csv", index=False)
    if "allocations" not in written:
        pd.DataFrame(columns=["seed", "policy", "day", "county_index", "fips", "cost"]).to_csv(out / "allocations.csv", index=False)
    metadata = {"data_kind": "SYNTHETIC DEBUGGING ONLY" if args.synthetic else "user supplied empirical inputs (provenance must be reviewed)",
                "config": asdict(cfg), "days": args.days, "seeds": list(range(args.seeds)),
                "county_order": frame.fips.tolist(), "policies": args.policies,
                "numpy_version": np.__version__}
    metadata["package_versions"]={}
    for package in ("numpy","scipy","pandas","scikit-learn","matplotlib","torch"):
        try:
            metadata["package_versions"][package]=version(package)
        except PackageNotFoundError:
            pass
    if args.data:
        metadata["input_sha256"]={name:hashlib.sha256((args.data/name).read_bytes()).hexdigest()
                                  for name in ("counties.csv","mobility.csv")}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    # Fraction of oracle improvement recovered; undefined when oracle improvement is nonpositive.
    wide = metrics.pivot(index="seed", columns="policy", values="infections")
    if {"epidemic", "oracle", "predicted"}.issubset(wide.columns):
        denominator = wide.epidemic-wide.oracle
        pd.DataFrame({"oracle_advantage": denominator, "recovered_fraction":
            (wide.epidemic-wide.predicted)/denominator.where(denominator>1e-9)}).to_csv(out/"prediction_value.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grouped = metrics.groupby("policy").infections.mean()
    ax = grouped.plot.bar(ylabel="Mean incident infections during simulation")
    ax.set_title(metadata["data_kind"])
    plt.tight_layout()
    plt.savefig(out / "infections.png", dpi=150)
    plt.close()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
