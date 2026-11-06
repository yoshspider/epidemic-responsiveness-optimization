"""Download and prepare the county inputs used by the historical replay."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy import sparse

RAW = Path("data/raw")
TRANSFORMED = Path("data/transformed")
WEEKS = pd.date_range("2020-09-07", periods=12, freq="7D")
SOURCES = {
    "nyt_cases_2020.csv": "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2020.csv",
    "nyt_mask_use.csv": "https://raw.githubusercontent.com/nytimes/covid-19-data/master/mask-use/mask-use-by-county.csv",
    "census_co_est2019.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv",
    "alternative_sources/cdc_svi2018_counties.csv": (
        "https://svi2.cdc.gov/webapi/Documents/download?year=2018&type=csv"
        "&category=states_counties&name=SVI_2018_US_COUNTY"
    ),
}


def fips(series):
    raw = series.astype(str).str.strip()
    if not raw.str.fullmatch(r"\d{1,5}").all():
        raise ValueError("FIPS must be one to five digits (read as text)")
    return raw.str.zfill(5)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    next_progress = 256*1024*1024
    with urlopen(Request(url, headers={"User-Agent":"EpidemicResearch/0.1"}), timeout=120) as response, temp.open("wb") as out:
        while block := response.read(1024*1024):
            size += len(block)
            digest.update(block)
            out.write(block)
            if size>=next_progress:
                print(f"{destination.name}: {size/(1024**3):.2f} GiB downloaded",flush=True)
                next_progress+=256*1024*1024
    temp.replace(destination)
    return {"url":url,"path":str(destination),"bytes":size,"sha256":digest.hexdigest(),
            "retrieved_utc":datetime.now(timezone.utc).isoformat()}


def nyt_daily(frame):
    frame=frame.copy()
    audit={"raw_rows":len(frame),"missing_fips_rows":int(frame.fips.isna().sum())}
    frame=frame[frame.fips.notna()].copy()
    frame["fips"]=fips(frame.fips)
    frame["date"]=pd.to_datetime(frame.date,errors="raise")
    if frame.duplicated(["fips","date"]).any():
        raise ValueError("Duplicate NYT county dates")
    frame=frame.sort_values(["fips","date"])
    delta=frame.groupby("fips").cases.diff()
    delta=delta.fillna(frame.cases)
    gaps=frame.groupby("fips").date.diff().dt.days
    frame["reported_incidence"]=delta
    frame["negative_revision"]=delta<0
    frame["date_gap"]=gaps.fillna(1)!=1
    audit["negative_revision_rows"]=int(frame.negative_revision.sum())
    audit["non_daily_gap_rows"]=int(frame.date_gap.sum())
    # Negative reports remain flagged; no silent clipping or inferred infections.
    return frame,audit


def census_population(frame, year=2019):
    frame=frame[frame.SUMLEV.astype(str).str.zfill(3)=="050"].copy()
    frame["fips"]=frame.STATE.astype(str).str.zfill(2)+frame.COUNTY.astype(str).str.zfill(3)
    frame["population"]=pd.to_numeric(frame[f"POPESTIMATE{year}"],errors="raise")
    if frame.fips.duplicated().any() or (frame.population<=0).any():
        raise ValueError("Invalid Census populations")
    return frame[["fips","STNAME","CTYNAME","population"]]


def download_mobility(date):
    stamp = date.strftime('%Y_%m_%d')
    folder = RAW / "midterm_mobility"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(stamp+'.csv')
    meta = path.with_suffix('.json')
    if path.exists():
        return
    url = f'https://raw.githubusercontent.com/GeoDS/COVID19USFlows-WeeklyFlows/master/weekly_flows/county2county/weekly_county2county_{stamp}.csv'
    record = download(url, path)
    check = pd.read_csv(path, nrows=2)
    if not {'geoid_o','geoid_d','pop_flows'}.issubset(check):
        raise ValueError('Unexpected mobility source schema')
    meta.write_text(json.dumps(record, indent=2))
    print(f'Weekly mobility {stamp}: {record["bytes"]:,} bytes', flush=True)


def national_matrices(counties, out):
    n = len(counties)
    index = dict(zip(counties.fips, range(n)))
    matrices = []
    for week, date in enumerate(WEEKS):
        path = out / f'mobility_{week}.npz'
        if path.exists():
            matrix = sparse.load_npz(path)
        else:
            raw = pd.read_csv(
                RAW / f"midterm_mobility/{date:%Y_%m_%d}.csv", dtype={'geoid_o': str, 'geoid_d': str})
            for col in ['geoid_o', 'geoid_d']:
                raw[col] = raw[col].str.zfill(5)
            raw = raw[raw.geoid_o.isin(index) & raw.geoid_d.isin(index)
                      & raw.geoid_o.ne(raw.geoid_d)]
            matrix = sparse.coo_matrix(
                (raw.pop_flows.to_numpy(float), (raw.geoid_d.map(index), raw.geoid_o.map(index))), shape=(
                    n, n)).tocsr()
            row = np.asarray(matrix.sum(1)).ravel()
            matrix = sparse.diags(np.divide(1, row, out=np.zeros_like(row), where=row > 0)) @ matrix
            sparse.save_npz(path, matrix)
        matrices.append(matrix)
    return matrices


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    TRANSFORMED.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, url in SOURCES.items():
        destination = RAW / name
        if not destination.exists():
            provenance.append(download(url, destination))
    if provenance:
        audit_path = RAW / "bootstrap_downloads.json"
        previous = json.loads(audit_path.read_text()) if audit_path.exists() else []
        audit_path.write_text(json.dumps(previous + provenance, indent=2))

    population = census_population(pd.read_csv(RAW / "census_co_est2019.csv", encoding="latin-1"))
    population.to_csv(TRANSFORMED / "census_population_2019.csv", index=False)
    reports, audit = nyt_daily(pd.read_csv(RAW / "nyt_cases_2020.csv", dtype={"fips": str}))
    reports.to_csv(TRANSFORMED / "nyt_daily_reports.csv", index=False)
    (TRANSFORMED / "nyt_bootstrap_audit.json").write_text(json.dumps(audit, indent=2))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download_mobility, WEEKS))
    print("County inputs ready for the 12-week replay.")


if __name__ == "__main__":
    main()
