"""Describe candidate score variation and coverage on the confirmed geography."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .us_weekly_proxy import OUT, SOURCE


def main():
    county = pd.read_csv(SOURCE / "counties.csv", dtype={"fips": str})
    surveys = pd.read_csv(SOURCE / "proxy_values.csv", dtype={"fips": str}).set_index("fips")
    ctis = pd.read_csv("data/transformed/ctis_october2020_monthly_proxy.csv", dtype={"fips": str})
    ctis = ctis.loc[(ctis.month == "2020-10") & (ctis.observed_days >= 7)].set_index("fips")
    daily = pd.read_csv(Path("data/raw/vaccine_experiment/acceptance_2021-02.csv"),
                        dtype={"geo_value": str})
    keep = daily.time_value.between(20210201, 20210228) & (daily.issue < 20210526)
    daily = (daily.loc[keep].sort_values("issue")
             .drop_duplicates(["geo_value", "time_value"], keep="last"))
    acceptance = daily.groupby("geo_value").value.agg(["mean", "count"])
    data = county.set_index("fips").join(surveys)
    data["ctis_mask"] = ctis.responsiveness
    data["ctis_acceptance"] = acceptance["mean"].where(acceptance["count"] >= 7) / 100
    rows = []
    for scope, frame in [
        ("available", data), ("common", data.dropna()),
    ]:
        for name in ("mask", "ctis_mask", "ctis_acceptance", "vaccine"):
            match = frame.dropna(subset=[name])
            x = match[name]
            w = match.population
            mean = np.average(x, weights=w)
            rows.append(dict(scope=scope, indicator=name, counties=len(x),
                             population_share=w.sum()/data.population.sum(),
                             county_mean=x.mean(), county_sd=x.std(ddof=0),
                             population_weighted_sd=np.sqrt(np.average((x - mean)**2, weights=w)),
                             p05=x.quantile(.05), p95=x.quantile(.95),
                             spearman_with_hhs=spearmanr(x, match.vaccine).statistic))
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "candidate_variation.csv", index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
