from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from .model import State


def fips(series):
    raw = series.astype(str).str.strip()
    if not raw.str.fullmatch(r"\d{1,5}").all():
        raise ValueError("FIPS must be one to five digits (read as text)")
    return raw.str.zfill(5)


def load_data(folder, start_date="2020-07-06"):
    folder = Path(folder)
    counties = pd.read_csv(folder / "counties.csv", dtype={"fips": str})
    counties["fips"] = fips(counties.fips)
    if counties.fips.duplicated().any():
        raise ValueError("Duplicate county FIPS")
    required = ["population_general", "population_vulnerable", "responsiveness",
                "cost_per_person", "infected_general", "infected_vulnerable",
                "recovered_general", "recovered_vulnerable"]
    if not set(required).issubset(counties.columns):
        raise ValueError(f"Required county columns: {required}")
    values = counties[required].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite county data; resolve explicitly before simulation")
    n = values[:, :2]
    i, r = values[:, 4:6], values[:, 6:8]
    edges = pd.read_csv(folder / "mobility.csv", dtype={"source_fips": str, "destination_fips": str})
    for col in ("source_fips", "destination_fips"):
        edges[col] = fips(edges[col])
    index = {f: j for j, f in enumerate(counties.fips)}
    if not set(edges.source_fips).union(edges.destination_fips).issubset(index):
        raise ValueError("Mobility refers to counties absent from the county table")
    def matrix(table):
        return sparse.coo_matrix((table.weight.to_numpy(float),
                (table.destination_fips.map(index), table.source_fips.map(index))), shape=(len(n), len(n)))
    if "week_start" in edges:
        dates = pd.to_datetime(edges.week_start, errors="raise")
        if dates.isna().any():
            raise ValueError("Missing mobility week dates")
        edges["offset"] = (dates-pd.Timestamp(start_date)).dt.days
        mobility = {float(day): matrix(group) for day, group in edges.groupby("offset")}
    else:
        mobility = matrix(edges)
    predicted = counties.predicted_responsiveness.to_numpy(float) if "predicted_responsiveness" in counties else None
    return counties, n, values[:, 2], values[:, 3], mobility, State(n - i - r, i, r, np.zeros(len(n))), predicted


def synthetic(n=24, seed=123):
    """Controlled debugging data, NEVER empirical county findings."""
    rng = np.random.default_rng(seed)
    total = rng.integers(10000, 100000, n)
    vulnerable = np.rint(total * rng.uniform(.1, .3, n))
    population = np.column_stack((total - vulnerable, vulnerable))
    infected = np.maximum(1, np.rint(population * rng.uniform(.001, .01, (n, 1))))
    p = rng.uniform(.3, .95, n)
    costs = rng.uniform(.01, .08, n)
    edges = sparse.coo_matrix((np.ones(2*n),
            (np.repeat(np.arange(n), 2), np.column_stack(((np.arange(n)+1)%n, (np.arange(n)-1)%n)).ravel())), shape=(n,n))
    frame = pd.DataFrame({"fips": [f"{j+1:05d}" for j in range(n)]})
    return frame, population, p, costs, edges, State(population-infected, infected, np.zeros_like(population), np.zeros(n)), None
