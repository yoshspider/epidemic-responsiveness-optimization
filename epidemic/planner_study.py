"""Compare stronger planning and normalized national budgets without hidden rescaling."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .model import Config,Model
from .observed_study import prepare
from .experiments import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("outputs/planner_study"))
    p.add_argument("--days",type=int,default=84)
    p.add_argument("--horizon",type=int,default=28)
    p.add_argument("--iterations",type=int,default=8)
    p.add_argument("--fraction",type=float,default=.01)
    p.add_argument("--mpc",action="store_true")
    p.add_argument("--seed",type=int,default=-1,help="-1 deterministic; nonnegative paired stochastic seed")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    counties,n,_,initial,mobility,_,audit=prepare(Path("data/transformed"))
    fit=json.loads(Path("outputs/observed_study/calibration.json").read_text())
    response=counties.nyt_mask_comparison.to_numpy()
    config=Config(beta_local=fit["beta_local"],beta_mobility=fit["beta_mobility"],
                  prior=float(response.mean()),mean_prior=float(response.mean()),budget=float(n.sum()*a.fraction),
                  horizon_days=a.horizon,planning_iterations=a.iterations,stochastic=a.seed>=0,start_date="2020-08-31")
    model=Model(n,response,np.ones(len(n)),mobility,config)
    manifest={"config":asdict(config),"days":a.days,"fraction":a.fraction,
              "data_kind":audit["data_kind"],"input_audit":audit,"seed":a.seed,
              "budget_units":"person-equivalent effort per week, not dollars; explicitly normalized by retained population",
              "policies":["none","epidemic","oracle"]+(["mpc_epidemic","mpc"] if a.mpc else [])}
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    metrics=pd.read_csv(a.output/'metrics.csv').to_dict('records') if a.resume and (a.output/'metrics.csv').exists() else []
    completed={row['policy'] for row in metrics}
    for policy in manifest["policies"]:
        if policy in completed:
            continue
        start=time.perf_counter()
        m,logs,trajectory,allocation=run(model,initial,policy,a.days,max(a.seed,0),record=True)
        m.pop("vulnerable")
        m.pop("deaths")
        m["runtime_seconds"]=time.perf_counter()-start
        metrics.append(m)
        pd.DataFrame(metrics).to_csv(a.output/"metrics.csv",index=False)
        # Weekly snapshot data for spatial comparisons; incident sums count every daily step.
        traj=pd.DataFrame(trajectory)
        mapping=dict(enumerate(counties.fips))
        traj["fips"]=traj.county_index.map(mapping)
        traj["week"]=np.floor((traj.day-1)/7).astype(int)+1
        weekly=traj.groupby(["week","fips"],sort=True).agg(incidence=("new_general","sum"),
                   active=("infected","last"),susceptible=("susceptible","last"),effort=("effort","last")).reset_index()
        weekly.to_csv(a.output/(policy+"_weekly.csv"),index=False)
        pd.DataFrame({"fips":counties.fips,"person_equivalent_effort":allocation,"cost":allocation*model.cost}).to_csv(a.output/(policy+"_allocation.csv"),index=False)
        if logs:
            table=pd.DataFrame(logs)
            table["fips"]=table.county_index.map(mapping)
            table.to_csv(a.output/(policy+"_decisions.csv"),index=False)
        print(f"{policy}: {m['infections']:,.0f} infections, {m['runtime_seconds']:.1f}s",flush=True)


if __name__=="__main__":
    main()
