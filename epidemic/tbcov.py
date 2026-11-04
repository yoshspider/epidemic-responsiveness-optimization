"""Stream base-release labels to county/month features without hydration."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import fips


def aggregate(path, crosswalk, chunksize=100000, location="user_loc",progress=False):
    crosswalk=crosswalk.copy()
    crosswalk["fips"]=fips(crosswalk.fips)
    for col in ("state","county"):
        crosswalk[col]=crosswalk[col].str.strip().str.casefold()
    if crosswalk.duplicated(["state","county"]).any():
        raise ValueError("Ambiguous state/county crosswalk")
    columns=["date_time","sentiment_label","sentiment_conf",location+"_country_code",location+"_state",location+"_county"]
    totals=None
    audit={"rows_read":0,"us_rows":0,"mapped_rows":0,"invalid_sentiment":0,
           "first_timestamp":None,"last_timestamp":None,
           "location_field":location,"county_rule":"exact casefolded state and county match; no fuzzy assignment"}
    for chunk in pd.read_csv(path,sep="\t",usecols=columns,dtype=str,chunksize=chunksize):
        audit["rows_read"]+=len(chunk)
        if progress and audit['rows_read']%1000000==0:
            print(f"Read {audit['rows_read']:,} base records; mapped {audit['mapped_rows']:,} US county records",flush=True)
        chunk=chunk[chunk[location+"_country_code"].eq("US")].copy()
        audit["us_rows"]+=len(chunk)
        chunk["state"]=chunk[location+"_state"].str.strip().str.casefold()
        chunk["county"]=chunk[location+"_county"].str.strip().str.casefold()
        chunk=chunk.merge(crosswalk[["state","county","fips"]],on=["state","county"],how="inner",validate="many_to_one")
        audit["mapped_rows"]+=len(chunk)
        chunk["sentiment"]=pd.to_numeric(chunk.sentiment_label,errors="coerce")
        valid=chunk.sentiment.isin([-1,0,1])
        audit["invalid_sentiment"]+=int((~valid).sum())
        chunk=chunk[valid].copy()
        if chunk.empty:
            continue
        dates=pd.to_datetime(chunk.date_time,utc=True,errors="raise")
        if dates.isna().any():
            raise ValueError("Missing tweet timestamp")
        first,last=dates.min().isoformat(),dates.max().isoformat()
        audit['first_timestamp']=min(audit['first_timestamp'],first) if audit['first_timestamp'] else first
        audit['last_timestamp']=max(audit['last_timestamp'],last) if audit['last_timestamp'] else last
        chunk["month"]=dates.dt.strftime("%Y-%m")
        chunk["positive"]=(chunk.sentiment==1).astype(int)
        chunk["negative"]=(chunk.sentiment==-1).astype(int)
        piece=chunk.groupby(["fips","month"]).agg(post_count=("sentiment","size"),
                sentiment_sum=("sentiment","sum"),positive=("positive","sum"),negative=("negative","sum"))
        totals=piece if totals is None else totals.add(piece,fill_value=0)
    if totals is None:
        raise ValueError("No mapped valid sentiment records")
    result=totals
    for column in ("post_count","sentiment_sum","positive","negative"):
        result[column]=result[column].astype('int64')
    result["mean_sentiment"]=result.sentiment_sum/result.post_count
    result["positive_share"]=result.positive/result.post_count
    result["negative_share"]=result.negative/result.post_count
    return result.reset_index(),audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument("tsv",type=Path)
    p.add_argument("crosswalk",type=Path,help="CSV with fips,state,county; source and vintage documented")
    p.add_argument("--location",choices=["user_loc","geo","place"],default="user_loc")
    p.add_argument("--output",type=Path,default=Path("outputs/tbcov_county_month.csv"))
    a=p.parse_args()
    result,audit=aggregate(a.tsv,pd.read_csv(a.crosswalk,dtype=str),location=a.location)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    a.output.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
