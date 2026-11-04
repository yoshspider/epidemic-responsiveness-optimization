"""Controlled one-at-a-time mechanism checks; not empirical estimates."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from .data import synthetic, load_data
from .model import Config, Model
from .experiments import run, paired_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--days", type=int, default=28)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("outputs/sensitivity"))
    a = p.parse_args()
    if bool(a.data) == a.synthetic or a.days<1 or a.seeds<1:
        p.error("Choose --data or --synthetic and positive days/seeds")
    _, n, true_p, cost, mobility, initial, _ = load_data(a.data) if a.data else synthetic()
    base = Config(budget=1000, stochastic=True)
    rng = np.random.default_rng(12)
    scenarios = [("baseline",base,true_p)]
    for field, values in {"budget":[0,500,2000], "efficacy":[0,.2,.8],
                          "beta_mobility":[0,.1], "horizon_days":[7,14],
                          "objective":["vulnerable","deaths"]}.items():
        scenarios.extend((f"{field}={value}",replace(base,**{field:value}),true_p) for value in values)
    for factor in (0.,.5,1.5):
        modified=np.clip(true_p.mean()+factor*(true_p-true_p.mean()),0,1)
        scenarios.append((f"counterfactual_p_variance_scale={factor}",base,modified))
    risk=(initial.s.sum(1)*initial.i.sum(1)/n.sum(1))/cost
    order=np.argsort(risk)
    for label, ordered_p in (("positive",np.sort(true_p)),("negative",np.sort(true_p)[::-1])):
        modified=np.empty_like(true_p)
        modified[order]=ordered_p
        scenarios.append((f"counterfactual_p_risk_correlation={label}",base,modified))
    scenarios.append(("counterfactual_permuted_p",base,rng.permutation(true_p)))
    metrics=[]
    for name,cfg,responsiveness in scenarios:
        model=Model(n,responsiveness,cost,mobility,cfg)
        print(name,flush=True)
        for seed in range(a.seeds):
            for policy in ("none","epidemic","oracle","lookahead"):
                m,*_=run(model,initial,policy,a.days,seed)
                metrics.append({"scenario":name,**m})
    # Same true environment; prediction noise is a separate information perturbation.
    model=Model(n,true_p,cost,mobility,base)
    for sigma in (0.,.05,.15):
        noise=np.random.default_rng(77).normal(0,sigma,len(n))
        estimate=np.clip(true_p+noise,0,1)
        for seed in range(a.seeds):
            for policy in ("epidemic","oracle","predicted"):
                m,*_=run(model,initial,policy,a.days,seed,estimate)
                metrics.append({"scenario":f"counterfactual_prediction_noise={sigma}",**m})
    a.output.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(metrics)
    table.to_csv(a.output/"metrics.csv",index=False)
    contrasts=[]
    for name,group in table.groupby("scenario"):
        summary=paired_summary(group)
        summary["scenario"]=name
        contrasts.append(summary)
    pd.concat(contrasts).to_csv(a.output/"paired_comparisons.csv",index=False)
    (a.output/"DATA_KIND.txt").write_text("Controlled synthetic mechanism checks" if a.synthetic else
        "User-supplied inputs with explicitly counterfactual responsiveness/prediction scenarios. Not historical outcomes.")


if __name__ == "__main__":
    main()
