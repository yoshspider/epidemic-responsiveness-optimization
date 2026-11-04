"""National, empirical-input counterfactual study with explicit observation/cost assumptions.

Not an empirical estimate of causal intervention efficacy; not CTIS/TBCOV prediction.
Uses fixed July 6 mobility snapshot, July NYT mask proxy, historical Census totals.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
from .model import Model, Config, State
from .calibration import calibrate
from .experiments import run


def prepare(folder, reporting_fraction=.2):
    population=pd.read_csv(folder/"census_population_2019.csv",dtype={"fips":str})
    mask=pd.read_csv(folder/"nyt_mask_independent_comparison.csv",dtype={"fips":str})
    cases=pd.read_csv(folder/"nyt_daily_reports.csv",dtype={"fips":str},parse_dates=["date"])
    counties=population.merge(mask,on="fips",validate="one_to_one").sort_values("fips").reset_index(drop=True)
    counts=cases.pivot(index="date",columns="fips",values="cases").reindex(columns=counties.fips)
    daily=cases.pivot(index="date",columns="fips",values="reported_incidence").reindex(columns=counties.fips)
    gap=cases.pivot(index="date",columns="fips",values="date_gap").reindex(columns=counties.fips).fillna(True)
    def state_on(date):
        previous=pd.Timestamp(date)-pd.Timedelta(days=1)
        old=previous-pd.Timedelta(days=7)
        total=counts.loc[previous].to_numpy()/reporting_fraction
        infected=(counts.loc[previous]-counts.loc[old]).to_numpy()/reporting_fraction
        removed=total-infected
        pop=counties.population.to_numpy(float)
        valid=np.isfinite(infected)&np.isfinite(removed)&(infected>=0)&(removed>=0)&(pop-total>=0)
        return infected,removed,valid
    train_i,train_r,train_valid=state_on("2020-07-20")
    eval_i,eval_r,eval_valid=state_on("2020-08-31")
    keep=train_valid&eval_valid
    exclusions=counties.loc[~keep,"fips"].tolist()
    counties=counties.loc[keep].reset_index(drop=True)
    n=np.column_stack((counties.population.to_numpy(float),np.zeros(len(counties))))
    def initial(infected,removed):
        i=np.column_stack((np.rint(infected[keep]),np.zeros(len(counties))))
        r=np.column_stack((np.rint(removed[keep]),np.zeros(len(counties))))
        return State(n-i-r,i,r,np.zeros(len(counties)))
    training=initial(train_i,train_r)
    evaluation=initial(eval_i,eval_r)
    dates=pd.date_range("2020-07-20","2020-08-30")
    reports=daily.loc[dates].to_numpy()[:,keep]
    gap_flags=gap.loc[dates].to_numpy(dtype=bool)[:,keep]
    exclusions_count=int(((reports<0)|gap_flags|~np.isfinite(reports)).sum())
    reports[(reports<0)|gap_flags]=np.nan
    edges=pd.read_csv(folder/"mobility_20200706.csv",dtype={"source_fips":str,"destination_fips":str})
    index=dict(zip(counties.fips,np.arange(len(counties))))
    inside=edges.source_fips.isin(index)&edges.destination_fips.isin(index)
    original_weight=float(edges.weight.sum())
    retained=edges.loc[inside]
    matrix=sparse.coo_matrix((retained.weight.to_numpy(),(retained.destination_fips.map(index),retained.source_fips.map(index))),shape=(len(counties),len(counties)))
    audit={"data_kind":"EMPIRICAL-INPUT COUNTERFACTUAL, NOT CAUSAL EFFECT ESTIMATION",
           "counties":len(counties),"excluded_counties":exclusions,"population":float(n.sum()),
           "reporting_fraction":reporting_fraction,"initialization":"active = last seven days cumulative-report difference / fixed reporting fraction; removed = earlier cumulative reports / fraction; counts rounded",
           "training_start":"2020-07-20","training_end":"2020-08-16","holdout_start":"2020-08-17","holdout_end":"2020-08-30",
           "policy_start":"2020-08-31","mobility":"July 6, 2020 snapshot held fixed, not actual weekly dynamics",
           "retained_mobility_edges":len(retained),"original_edges":len(edges),
           "excluded_flow_weight_fraction":1-float(retained.weight.sum())/original_weight,
           "excluded_training_report_cells":exclusions_count,
           "cost":"assumed one model unit per person-equivalent effort everywhere; no dollar claim",
           "age_groups":"total population only; vulnerable compartment zero because age counts unavailable; no demographic death conclusions",
           "behavior":"NYT FREQUENTLY + ALWAYS proxy, not CTIS-trained responsiveness or causal response",
           "mask_mean":float(counties.nyt_mask_comparison.mean()),"mask_sd":float(counties.nyt_mask_comparison.std()),
           "source_sha256":{f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in
                            ("census_population_2019.csv","nyt_mask_independent_comparison.csv","nyt_daily_reports.csv","mobility_20200706.csv")}}
    return counties,n,training,evaluation,matrix,reports,audit


def summarize(table):
    summaries=[]
    for scenario,group in table.groupby("scenario"):
        wide=group.pivot(index="seed",columns="policy",values="infections")
        for comparator in ("epidemic","epidemic_mean"):
            delta=wide[comparator]-wide.oracle
            half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
            summaries.append({"scenario":scenario,"comparator":comparator,"mean_gain":delta.mean(),
                "ci_low":delta.mean()-half,"ci_high":delta.mean()+half,
                "relative_gain_percent":100*delta.mean()/wide[comparator].mean(),
                "seeds":len(delta),"oracle_win_fraction":float((delta>0).mean())})
    return pd.DataFrame(summaries)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--transformed",type=Path,default=Path("data/transformed"))
    p.add_argument("--output",type=Path,default=Path("outputs/observed_study"))
    p.add_argument("--days",type=int,default=84)
    p.add_argument("--seeds",type=int,default=10)
    p.add_argument("--reporting-fraction",type=float,default=.2)
    p.add_argument("--phase",choices=["screen","confirm"],default="screen")
    a=p.parse_args()
    if a.days<1 or a.seeds<1 or not 0<a.reporting_fraction<=1:
        p.error("Positive days/seeds and reporting fraction in (0,1] required")
    a.output.mkdir(parents=True,exist_ok=True)
    counties,n,train_initial,initial,mobility,reports,audit=prepare(a.transformed,a.reporting_fraction)
    response=counties.nyt_mask_comparison.to_numpy()
    base=Config(prior=.776,mean_prior=float(response.mean()),budget=100000,stochastic=False,start_date="2020-08-31")
    model=Model(n,response,np.ones(len(n)),mobility,base)
    fit_path=a.output/"calibration.json"
    if fit_path.exists():
        fit=json.loads(fit_path.read_text())
        # Reuse only if the explicit observation assumptions and source hashes match.
        previous=json.loads((a.output/"input_audit.json").read_text())
        if previous["source_sha256"]!=audit["source_sha256"] or fit["reporting_fraction"]!=a.reporting_fraction:
            raise ValueError("Existing calibration belongs to different inputs")
    else:
        print(f"Calibrating {len(n)} counties: 28 training days, 14 holdout days",flush=True)
        fit,prediction=calibrate(model,train_initial,reports,28,a.reporting_fraction)
        fit_path.write_text(json.dumps(fit,indent=2))
        observed_national=np.nansum(reports,axis=1)
        predicted_national=prediction.sum(1)
        pd.DataFrame({"date":pd.date_range("2020-07-20","2020-08-30"),"observed_reports_excluded_revisions":observed_national,
                      "predicted_reports":predicted_national,"partition":["train"]*28+["holdout"]*14}).to_csv(a.output/"calibration_series.csv",index=False)
        # Compare on exactly observed cells; include a simple last-seven-day county forecast.
        test=reports[28:]
        valid=np.isfinite(test)
        persistence=np.nanmean(reports[21:28],axis=0)
        fitted_error=(prediction[28:][valid]-test[valid])**2
        naive_error=(np.broadcast_to(persistence,test.shape)[valid]-test[valid])**2
        fit["naive_holdout_rmse"]=float(np.sqrt(np.mean(naive_error)))
        fit["holdout_rmse_vs_naive_ratio"]=float(np.sqrt(fitted_error.mean()/naive_error.mean()))
        fit_path.write_text(json.dumps(fit,indent=2))
    (a.output/"input_audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(fit),flush=True)
    base.beta_local=fit["beta_local"]
    base.beta_mobility=fit["beta_mobility"]
    base.stochastic=a.phase=="confirm"
    scenarios=[("baseline",{}),("budget_low",{"budget":25000}),
               ("budget_high",{"budget":1000000}),("efficacy_low",{"efficacy":.2}),
               ("efficacy_high",{"efficacy":.8}),("chunk_small",{"chunk":5000}),
               ("persistent",{"retention_daily":1.}),
               ("no_mobility",{"beta_mobility":0.})]
    metadata={"base_config":asdict(base),"phase":a.phase,"days":a.days,"seeds":a.seeds if base.stochastic else 1,
              "scenarios":scenarios,"calibration":fit,"data_kind":audit["data_kind"],
              "mean_prior_information":"epidemic_mean is explicitly given aggregate mask mean; epidemic uses earlier .776 prior"}
    (a.output/(a.phase+"_manifest.json")).write_text(json.dumps(metadata,indent=2))
    rows=[]
    for name,changes in scenarios:
        cfg=Config(**{**asdict(base),**changes})
        model=Model(n,response,np.ones(len(n)),mobility,cfg)
        for seed in range(a.seeds if base.stochastic else 1):
            for policy in ("none","epidemic","epidemic_mean","oracle"):
                m,_,_,allocation=run(model,initial,policy,a.days,seed+5000,record=False)
                # Suppress placeholder vulnerable/death metrics in this age-unavailable study.
                m.pop("vulnerable")
                m.pop("deaths")
                rows.append({"scenario":name,**m,"selected_counties":int((allocation>0).sum()),
                             "person_equivalent_effort":float(allocation.sum())})
        pd.DataFrame(rows).to_csv(a.output/(a.phase+"_trials.csv"),index=False)
        print(f"{a.phase}: {name} complete",flush=True)
    summarize(pd.DataFrame(rows)).to_csv(a.output/(a.phase+"_summary.csv"),index=False)


if __name__=="__main__":
    main()
