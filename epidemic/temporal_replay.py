"""Retrospective weekly-transmission replay fitted to actual NYT reports.

Each week's rates are fitted using that week's outcomes. This is explicitly
retrospective, NOT a prospective forecasting evaluation. All policies share the
same realized rate schedule and observe only the current rates in planning.
"""
import argparse
import json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .observed_study import prepare
from .model import Model,Config
from .experiments import run


def fit_schedule(n,response,mobility,initial,target,mobility_share=None):
    state=initial.copy()
    base=Config(beta_local=.1131043293,beta_mobility=.0220848209)
    history=[]
    predictions=[]
    for start in range(0,len(target),7):
        observations=target[start:start+7]
        valid=np.isfinite(observations)
        def simulate(rates):
            model=Model(n,response,np.ones(len(n)),mobility,replace(base,beta_local=rates[0],beta_mobility=rates[1]))
            current=state.copy()
            values=[]
            for _ in range(len(observations)):
                current,new=model.step(current)
                values.append(new.sum(1)*.2)
            return np.asarray(values),current
        def residual(rates):
            prediction,_=simulate(rates)
            return np.sqrt(prediction[valid]+.375)-np.sqrt(observations[valid]+.375)
        if mobility_share is None:
            fit=least_squares(residual,[base.beta_local,base.beta_mobility],bounds=(0,1),max_nfev=80)
            estimated=fit.x
        else:
            def constrained(value):
                return residual(np.array([value[0]*(1-mobility_share),value[0]*mobility_share]))
            fit=least_squares(constrained,[base.beta_local+base.beta_mobility],bounds=(0,1.5),max_nfev=80)
            estimated=np.array([fit.x[0]*(1-mobility_share),fit.x[0]*mobility_share])
        prediction,state=simulate(estimated)
        base.beta_local,base.beta_mobility=estimated
        singular=np.linalg.svd(fit.jac,compute_uv=False)
        history.append({'day':start,'week':start//7+1,'beta_local':estimated[0],'beta_mobility':estimated[1],
                        'success':bool(fit.success),'in_sample_rmse':float(np.sqrt(np.mean((prediction[valid]-observations[valid])**2))),
                        'jacobian_condition':float(singular[0]/singular[-1]) if singular[-1]>0 else np.inf})
        predictions.append(prediction)
        print(f"Fitted retrospective week {start//7+1}: betaL={estimated[0]:.4f}, betaM={estimated[1]:.4f}",flush=True)
    return pd.DataFrame(history),np.vstack(predictions)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('outputs/temporal_replay'))
    p.add_argument('--seeds',type=int,default=0)
    p.add_argument('--mobility-share',type=float,help='Fix mobility share of total rate to test decomposition identifiability')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    counties,n,_,initial,mobility,_,audit=prepare(Path('data/transformed'))
    response=counties.nyt_mask_comparison.to_numpy()
    dates=pd.date_range('2020-08-31','2020-11-22')
    source=pd.read_csv('data/transformed/nyt_daily_reports.csv',dtype={'fips':str},parse_dates=['date'])
    source=source[source.fips.isin(counties.fips)]
    source['target']=source.reported_incidence.where((source.reported_incidence>=0)&(~source.date_gap))
    target=source.pivot(index='date',columns='fips',values='target').reindex(index=dates,columns=counties.fips).to_numpy()
    path=a.output/'weekly_rates.csv'
    if path.exists():
        rates=pd.read_csv(path)
    else:
        rates,prediction=fit_schedule(n,response,mobility,initial,target,a.mobility_share)
        rates.to_csv(path,index=False)
        pd.DataFrame({'date':dates,'observed_reports':np.nansum(target,axis=1),
                      'fitted_reports':np.where(np.isfinite(target),prediction,0).sum(1)}).to_csv(a.output/'replay_fit_series.csv',index=False)
    schedule=tuple(rates[['day','beta_local','beta_mobility']].itertuples(index=False,name=None))
    base=Config(prior=float(response.mean()),mean_prior=float(response.mean()),beta_schedule=schedule,
                stochastic=a.seeds>0,start_date='2020-08-31')
    manifest={'data_kind':'RETROSPECTIVE EMPIRICAL-INPUT COUNTERFACTUAL REPLAY',
              'fit':'weekly global betaL and betaM fitted to contemporaneous NYT cases; in-sample replay, not held-out forecast',
              'config':asdict(base),'input_audit':audit,'intervention_effects_and_costs':'still assumed, not causally estimated',
              'seeds':a.seeds if a.seeds else 1,'design_budget_fractions':[.005,.02,.10],
              'fixed_mobility_share':a.mobility_share}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    for fraction in (.005,.02,.10):
        model=Model(n,response,np.ones(len(n)),mobility,replace(base,budget=float(n.sum()*fraction)))
        for seed in range(a.seeds if a.seeds else 1):
            results={}
            for policy in ('none','epidemic','oracle'):
                m,*_=run(model,initial,policy,84,seed+8000,record=False)
                results[policy]=m['infections']
            records.append({'budget_fraction':fraction,'seed':seed,**results,
                'information_gain':results['epidemic']-results['oracle'],
                'relative_gain_percent':100*(results['epidemic']-results['oracle'])/results['epidemic']})
        pd.DataFrame(records).to_csv(a.output/'trials.csv',index=False)
        print(f"Retrospective replay budget fraction {fraction}: complete",flush=True)


if __name__=='__main__':
    main()
