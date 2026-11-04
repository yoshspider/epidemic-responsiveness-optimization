"""Fixed design: mean-matched behavior heterogeneity, cost/risk correlation, budget.

Confirmation uses fresh networks and fresh epidemic seeds, not extra draws on
the exploratory world. All scenarios are retained, including negative outcomes.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, t, spearmanr
from .data import synthetic
from .model import Model, Config
from .experiments import run


def behavior(n, mean, sd, risk, correlation, seed):
    if sd == 0:
        return np.full(n,mean)
    concentration=mean*(1-mean)/sd**2-1
    if concentration<=0:
        raise ValueError("Infeasible beta mean/variance")
    values=beta.ppf((np.arange(n)+.5)/n,mean*concentration,(1-mean)*concentration)
    # Correct finite quantile mean by a common shift, retaining ranks and bounds.
    low,high=-1.,1.
    for _ in range(60):
        shift=(low+high)/2
        if np.clip(values+shift,0,1).mean()<mean:
            low=shift
        else:
            high=shift
    values=np.clip(values+(low+high)/2,0,1)
    if correlation=="independent":
        return np.random.default_rng(seed).permutation(values)
    order=np.argsort(risk)
    result=np.empty(n)
    result[order]=values if correlation=="positive" else values[::-1]
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("outputs/controlled_study"))
    p.add_argument("--days",type=int,default=84)
    p.add_argument("--worlds",type=int,default=10)
    p.add_argument("--stage",choices=["screen","confirm"],default="screen")
    a=p.parse_args()
    if a.worlds<1 or a.days<1:
        p.error("Positive worlds and days required")
    a.output.mkdir(parents=True,exist_ok=True)
    base=Config(prior=.716,mean_prior=.716,budget=1000,stochastic=a.stage=="confirm")
    design=[{"sd":sd,"correlation":correlation,"budget_fraction":fraction}
            for sd in (0.,.05,.11,.20,.30)
            for correlation in ("independent","positive","negative")
            for fraction in (.002,.01,.05,.20)]
    manifest={"data_kind":"SYNTHETIC CONTROLLED MECHANISM EXPERIMENT", "stage":a.stage,
              "days":a.days,"worlds":a.worlds,"config":asdict(base),"design":design,
              "mean_target":.716,"world_seed_start":1000 if a.stage=="confirm" else 200,
              "budget_definition":"weekly fraction of cost of one population-equivalent effort across all counties",
              "uncertainty":"Confirm intervals span independently generated worlds and one paired epidemic seed per world; exploratory multiple comparisons, unadjusted CIs"}
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    records=[]
    for world in range(a.worlds):
        world_seed=manifest["world_seed_start"]+world
        _,n,_,cost,mobility,initial,_=synthetic(24,world_seed)
        full_cost=float(n.sum(1)@cost)
        proxy_model=Model(n,np.full(len(n),base.prior),cost,mobility,base)
        risk=(proxy_model.incidence(initial,np.full(len(n),base.prior))*proxy_model.weights).sum(1)/cost/n.sum(1)
        for scenario_id,scenario in enumerate(design):
            config=Config(**{**asdict(base),"budget":scenario["budget_fraction"]*full_cost})
            response=behavior(len(n),.716,scenario["sd"],risk,scenario["correlation"],world_seed+55)
            model=Model(n,response,cost,mobility,config)
            seed=2000+world if a.stage=="confirm" else 0
            metrics={}
            actions={}
            for policy in ("epidemic","oracle"):
                m,_,_,allocation=run(model,initial,policy,a.days,seed,record=False)
                metrics[policy]=m
                actions[policy]=allocation
            union=((actions["epidemic"]>0)|(actions["oracle"]>0)).sum()
            row={"world":world,"scenario_id":scenario_id,**scenario,"observed_p_mean":response.mean(),
                 "observed_p_sd":response.std(),"observed_risk_correlation":spearmanr(response,risk).statistic if scenario["sd"] else 0.,
                 "epidemic_infections":metrics["epidemic"]["infections"],"oracle_infections":metrics["oracle"]["infections"],
                 "prevented_by_county_information":metrics["epidemic"]["infections"]-metrics["oracle"]["infections"],
                 "relative_gain_percent":100*(metrics["epidemic"]["infections"]-metrics["oracle"]["infections"])/max(1,metrics["epidemic"]["infections"]),
                 "epidemic_spent":metrics["epidemic"]["spent"],"oracle_spent":metrics["oracle"]["spent"],
                 "jaccard":((actions["epidemic"]>0)&(actions["oracle"]>0)).sum()/union if union else 1.,
                 "allocation_l1":abs(actions["epidemic"]-actions["oracle"]).sum()}
            records.append(row)
        pd.DataFrame(records).to_csv(a.output/"trials.csv",index=False)
        print(f"{a.stage} world {world+1}/{a.worlds} complete ({len(design)} scenarios)",flush=True)
    table=pd.DataFrame(records)
    summaries=[]
    for (scenario_id,sd,correlation,budget),group in table.groupby(["scenario_id","sd","correlation","budget_fraction"]):
        delta=group.prevented_by_county_information
        half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
        summaries.append({"scenario_id":scenario_id,"sd":sd,"correlation":correlation,"budget_fraction":budget,
            "mean_gain":delta.mean(),"ci_low":delta.mean()-half,"ci_high":delta.mean()+half,
            "mean_relative_gain_percent":group.relative_gain_percent.mean(),"win_fraction":float((delta>0).mean()),
            "mean_jaccard":group.jaccard.mean(),"mean_allocation_l1":group.allocation_l1.mean(),"worlds":len(delta)})
    pd.DataFrame(summaries).to_csv(a.output/"summary.csv",index=False)


if __name__=="__main__":
    main()
