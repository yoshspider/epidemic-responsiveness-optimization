"""Source-specific transformations, preserving exclusions and assumptions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import fips


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


def geods(frame, week_start):
    result=frame.rename(columns={"geoid_o":"source_fips","geoid_d":"destination_fips","pop_flows":"weight"}).copy()
    for col in ("source_fips","destination_fips"):
        result[col]=fips(result[col])
    result["weight"]=pd.to_numeric(result.weight,errors="raise")
    if not np.isfinite(result.weight).all() or (result.weight<0).any():
        raise ValueError("Invalid flows")
    result["week_start"]=pd.Timestamp(week_start).strftime("%Y-%m-%d")
    return result[["source_fips","destination_fips","weight","week_start"]]


def ctis_monthly(frame):
    """Descriptive monthly average of daily smoothed estimates, NOT pooled survey estimator."""
    frame=frame.copy()
    frame["fips"]=fips(frame.geo_value)
    audit={"rows_read":len(frame)}
    excluded=frame.fips.str.endswith("000")
    audit["state_aggregate_rows_excluded"]=int(excluded.sum())
    frame=frame[~excluded].copy()
    if "missing_value" in frame:
        frame=frame[pd.to_numeric(frame.missing_value)==0]
    frame["date"]=pd.to_datetime(frame.time_value.astype(str),format="%Y%m%d",errors="raise")
    if frame.duplicated(["fips","date"]).any():
        raise ValueError("Select one CTIS signal and revision per county/date")
    frame["p"]=pd.to_numeric(frame.value)/100
    if not np.isfinite(frame.p).all() or not frame.p.between(0,1).all():
        raise ValueError("CTIS percent values invalid")
    frame["month"]=frame.date.dt.strftime("%Y-%m")
    output=frame.groupby(["fips","month"]).agg(responsiveness=("p","mean"),observed_days=("p","size"))
    return output.reset_index(),audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw",type=Path,default=Path("data/raw"))
    p.add_argument("--output",type=Path,default=Path("data/transformed"))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    audit={}
    daily,audit["nyt"]=nyt_daily(pd.read_csv(a.raw/"nyt_cases_2020.csv",dtype={"fips":str}))
    daily.to_csv(a.output/"nyt_daily_reports.csv",index=False)
    populations=census_population(pd.read_csv(a.raw/"census_co_est2019.csv",encoding="latin1",dtype={"STATE":str,"COUNTY":str,"SUMLEV":str}))
    populations.to_csv(a.output/"census_population_2019.csv",index=False)
    mask=pd.read_csv(a.raw/"nyt_mask_use.csv",dtype={"COUNTYFP":str})
    mask["fips"]=fips(mask.COUNTYFP)
    mask["nyt_mask_comparison"]=mask.FREQUENTLY+mask.ALWAYS
    mask[["fips","nyt_mask_comparison"]].to_csv(a.output/"nyt_mask_independent_comparison.csv",index=False)
    audit["census_counties"]=len(populations)
    audit["nyt_mask_counties"]=len(mask)
    audit["census_missing_mask"]=sorted(set(populations.fips)-set(mask.fips))
    audit["mask_missing_census"]=sorted(set(mask.fips)-set(populations.fips))
    audit["note"]="NYT mask retained for independent comparison; CTIS is intended training target. No cost or vulnerable counts invented."
    (a.output/"audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
