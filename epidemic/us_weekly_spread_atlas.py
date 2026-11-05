"""Export one comparable county-incidence plate per week for the main experiment."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
from scipy import sparse

from .us_perfect_figures import trajectory
from .us_proxy_figures import draw, geometry, mapping
from .us_weekly_proxy import OUT, SOURCE

POLICIES = {
    "none": "No intervention",
    "uninformed": "Optimized, no behavioral information",
    "population_only": "Population only",
    "mask_only": "July masking only",
    "mask_plus_population": "July masking + population",
    "acceptance_population_mean": "Early willingness + population",
    "surrogate_oracle": "Oracle: later county scores",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "spread")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counties = pd.read_csv(SOURCE / "counties.csv", dtype={"fips": str})
    matrices = [sparse.load_npz(SOURCE / f"mobility_{week}.npz") for week in range(12)]
    with np.load(SOURCE / "initial.npz") as initial:
        n, s, i = (initial[key] for key in ("n", "s", "i"))
    np.testing.assert_array_equal(counties.population, n)
    with np.load(OUT / "percentile_schedules.npz") as schedules, np.load(
        OUT / "acceptance_population_mean.npz"
    ) as acceptance:
        np.testing.assert_array_equal(counties.fips.to_numpy(dtype="U5"), schedules["fips"])
        truth = schedules["surrogate"]
        cumulative = {
            name: trajectory(n, s, i, truth, acceptance["dose"] if name ==
                             "acceptance_population_mean" else schedules[name], matrices)
            for name in POLICIES
        }
    expected = pd.read_csv(OUT / "summary.csv").query("scale == 'percentile'").set_index("policy")
    extra = pd.read_csv(OUT / "acceptance_followup.csv").set_index("policy")
    for name, data in cumulative.items():
        source = extra if name == "acceptance_population_mean" else expected
        np.testing.assert_allclose(data[-1].sum(), source.loc[name, "infections"], rtol=1e-9)
    weekly = {name: np.diff(path, axis=0, prepend=np.zeros((1, len(n))))
              for name, path in cumulative.items()}
    geo = geometry()
    ticks = [0, 10, 100, 1000, 10000, 40000]
    norm = SymLogNorm(linthresh=10, vmin=0, vmax=ticks[-1], clip=True)
    for week in range(12):
        fig, axes = plt.subplots(3, 3, figsize=(13, 9.5))
        for ax, (name, label) in zip(axes.flat, POLICIES.items()):
            cases = weekly[name][week]
            im = draw(ax, geo, mapping(counties, cases / n * 1e5), norm, "YlOrRd")
            ax.set_title(f"{label}\n{cases.sum() / 1e6:.2f}M new infections nationally",
                         fontsize=10, pad=5)
        for ax in list(axes.flat)[len(POLICIES):]:
            ax.axis("off")
        legend_ax = axes[2, 1]
        legend_ax.text(0, .68, "How to read the maps", weight="bold", fontsize=11)
        legend_ax.text(0, .51, "Darker counties had more new infections\nper resident during this week.",
                       fontsize=9.5, va="top")
        legend_ax.text(0, .19, "Compare the same colors across all policies\nand all 12 weeks.",
                       fontsize=9.5, va="top")
        fig.subplots_adjust(left=.025, right=.98, bottom=.14, top=.87, hspace=.29, wspace=.10)
        bar = fig.colorbar(im, cax=fig.add_axes([.23, .081, .54, .019]), orientation="horizontal")
        bar.locator = FixedLocator(ticks)
        bar.update_ticks()
        bar.set_ticklabels(["0", "10", "100", "1,000", "10,000", "40,000+"])
        bar.set_label("New modeled infections per 100,000 residents during the week")
        fig.suptitle(f"Week {week + 1:02d} | U.S. county spread by allocation policy",
                     fontsize=17, weight="bold", y=.968)
        fig.text(.5, .919, "Simulated outbreak on real U.S. geography. Later vaccine non-hesitancy "
                 "percentile stands in for responsiveness.", ha="center", fontsize=10)
        fig.text(.5, .019, "Intervention policies share the same weekly budget. "
                 "Maps use a common nonlinear color scale.", ha="center", fontsize=9)
        fig.savefig(args.output / f"week_{week + 1:02d}.png", dpi=118)
        plt.close(fig)
        print(f"Rendered week {week + 1:02d}", flush=True)


if __name__ == "__main__":
    main()
