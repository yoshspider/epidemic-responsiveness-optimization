"""Match planning horizon across policies; perturb information separately."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .controlled_study import behavior
from .data import synthetic
from .model import Model, Config
from .experiments import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("outputs/information_study"))
    p.add_argument("--worlds",type=int,default=8)
    p.add_argument("--days",type=int,default=84)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    base=Config(prior=.716,mean_prior=.716,horizon_days=14,stochastic=True)
    scenarios=[("constant",0.,"independent",.05),("reference",.11,"independent",.05),
               ("high_variance",.30,"independent",.20),("myopic_failure",.05,"negative",.20)]
    (a.output/"manifest.json").write_text(json.dumps({"data_kind":"SYNTHETIC CONFIRMATION, EXPLORATORY SELECTED MECHANISMS",
        "config":asdict(base),"world_seeds":list(range(3000,3000+a.worlds)),"scenarios":scenarios,
        "days":a.days,"selection":"Includes reference, null, largest screening-gain regime and negative screening-gain regime; fresh worlds; CIs unadjusted"},indent=2))
    rows=[]
    for world in range(a.worlds):
        _,n,_,cost,mobility,initial,_=synthetic(24,3000+world)
        risk=initial.s.sum(1)*initial.i.sum(1)/n.sum(1)**2/cost
        noise=np.random.default_rng(4000+world).normal(size=len(n))
        for name,sd,correlation,fraction in scenarios:
            response=behavior(len(n),.716,sd,risk,correlation,3010+world)
            cfg=Config(**{**asdict(base),"budget":fraction*float(n.sum(1)@cost)})
            model=Model(n,response,cost,mobility,cfg)
            policies=[("epidemic",None),("oracle",None),("lookahead_epidemic",None),("lookahead",None)]
            policies += [(f"prediction_noise_{sigma}",np.clip(response+sigma*noise,0,1)) for sigma in (.05,.15,.30)]
            for policy,estimate in policies:
                m,*_=run(model,initial,"predicted" if estimate is not None else policy,a.days,5000+world,estimate,record=False)
                if estimate is not None:
                    m["policy"]=policy
                    m["prediction_mae"]=float(abs(estimate-response).mean())
                rows.append({"world":world,"scenario":name,**m})
        pd.DataFrame(rows).to_csv(a.output/"trials.csv",index=False)
        print(f"information world {world+1}/{a.worlds} complete",flush=True)
    table=pd.DataFrame(rows)
    comparisons=[]
    for name,group in table.groupby("scenario"):
        wide=group.pivot(index="world",columns="policy",values="infections")
        for left,right in [("epidemic","oracle"),("lookahead_epidemic","lookahead"),
                           ("oracle","lookahead"),("epidemic","lookahead_epidemic")]+[("epidemic",f"prediction_noise_{sigma}") for sigma in (.05,.15,.30)]:
            delta=wide[left]-wide[right]
            half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
            denominator=(wide.epidemic-wide.oracle).mean()
            comparisons.append({"scenario":name,"left":left,"right":right,"mean_left_minus_right":delta.mean(),
                "ci_low":delta.mean()-half,"ci_high":delta.mean()+half,"relative_percent":100*delta.mean()/wide[left].mean(),
                "oracle_recovery_fraction":delta.mean()/denominator if right.startswith("prediction") and denominator>0 else np.nan})
    pd.DataFrame(comparisons).to_csv(a.output/"comparisons.csv",index=False)


if __name__=="__main__":
    main()
