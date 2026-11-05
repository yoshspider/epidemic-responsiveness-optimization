"""Plot weekly proxy outcomes and county differences from the uninformed plan."""

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from .us_perfect_figures import trajectory
from .us_proxy_figures import geometry, draw, mapping
from .us_weekly_proxy import OUT, SOURCE


def main():
    matrix = [sparse.load_npz(SOURCE / f"mobility_{w}.npz") for w in range(12)]
    with np.load(SOURCE / "initial.npz") as initial:
        n, s, i = (initial[key] for key in ("n", "s", "i"))
    names = {
        "none": "No intervention", "uninformed": "Uninformed",
        "mask_only": "Mask rank", "mask_plus_population": "Mask + population",
        "population_only": "Population only", "surrogate_oracle": "Surrogate oracle",
    }
    colors = ["#888888", "#273d57", "#de772d", "#089382", "#9574aa", "#166d9b"]
    with np.load(OUT / "percentile_schedules.npz") as plans:
        paths = {p: trajectory(n, s, i, plans["surrogate"], plans[p], matrix) for p in names}
        acceptance = OUT / "acceptance_population_mean.npz"
        if acceptance.exists():
            with np.load(acceptance) as extra:
                name = "acceptance_population_mean"
                paths[name] = trajectory(n, s, i, plans["surrogate"], extra["dose"], matrix)
                names[name] = "Early vaccine willingness + population"
                colors.append("#a04366")
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for (p, label), color in zip(names.items(), colors):
        total = paths[p].sum(axis=1) / 1e6
        ax.plot(range(1, 13), total, label=f"{label} ({total[-1]:.1f}M)",
                color=color, linewidth=2, linestyle="--" if p == "population_only" else "-")
    ax.set(xlabel="Week", ylabel="Cumulative modeled infections (millions)",
           title="U.S. weekly allocation: can prior surveys inform spending?")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.1, .015, "Simulated spread; vaccine non-hesitancy percentile stands in for responsiveness. "
             "Same budget for all intervention plans.", fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(OUT / "weekly_proxy_comparison.png", dpi=160)
    plt.close(fig)

    counties = pd.read_csv(SOURCE / "counties.csv", dtype={"fips": str})
    geo = geometry()
    key = "acceptance_population_mean" if acceptance.exists() else "mask_plus_population"
    benefit = (paths["uninformed"] - paths[key]) / n * 1e5
    limit = max(1, float(np.quantile(np.abs(benefit[[3, 7, 11]]), .99)))
    norm = SymLogNorm(linthresh=100, vmin=-limit, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, week in zip(axes, (4, 8, 12)):
        im = draw(ax, geo, mapping(counties, benefit[week - 1]), norm, "RdBu")
        ax.set_title(f"Week {week}")
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=.10, pad=.08,
                 shrink=.62, label="Cumulative difference per 100,000 (blue: fewer; red: more)")
    fig.suptitle(f"{names[key]} versus uninformed spending", fontsize=15)
    fig.savefig(OUT / "weekly_proxy_county_maps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
