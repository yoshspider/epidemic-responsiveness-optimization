"""Downstream evaluation of real out-of-fold TBCOV predictions on measured CTIS cohort.

Full county network, identical eligible treatment cohort for every policy. Outside
the cohort responsiveness is unidentified but irrelevant because effort is zero.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .observed_study import prepare
from .model import Config,Model,State
from .experiments import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prediction',type=Path,default=Path('outputs/tbcov_prediction'))
    p.add_argument('--output',type=Path,default=Path('outputs/tbcov_policy'))
    p.add_argument('--seeds',type=int,default=8)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    counties,n,_,_,mobility,_,source_audit=prepare(Path('data/transformed'))
    cases=pd.read_csv('data/transformed/nyt_daily_reports.csv',dtype={'fips':str},parse_dates=['date'])
    counts=cases.pivot(index='date',columns='fips',values='cases').reindex(columns=counties.fips)
    recent=counts.loc[pd.Timestamp('2020-11-01')].to_numpy()
    older=counts.loc[pd.Timestamp('2020-10-25')].to_numpy()
    infected=np.rint((recent-older)/.2)
    removed=np.rint(older/.2)
    valid=np.isfinite(infected)&np.isfinite(removed)&(infected>=0)&(removed>=0)&(n.sum(1)-infected-removed>=0)
    invalid=counties.loc[~valid,'fips'].tolist()
    counties=counties.loc[valid].reset_index(drop=True)
    n=n[valid]
    mobility=mobility.tocsr()[valid][:,valid]
    i=np.column_stack((infected[valid],np.zeros(len(n))))
    r=np.column_stack((removed[valid],np.zeros(len(n))))
    initial=State(n-i-r,i,r,np.zeros(len(n)))
    targets=pd.read_csv(a.prediction/'features_targets.csv',dtype={'fips':str}).set_index('fips')
    joined=targets.reindex(counties.fips)
    eligible=joined.responsiveness.notna().to_numpy()
    response=joined.responsiveness.fillna(0).to_numpy()
    # 0 outside cohort is a computational placeholder, never an imputed behavioral target.
    # No eligible policy can spend outside this cohort, so those values have no effect.
    rates=pd.read_csv('outputs/temporal_replay/weekly_rates.csv')
    rates['day']=rates.day-63
    rates=rates[rates.day>=0]
    schedule=tuple(rates[['day','beta_local','beta_mobility']].itertuples(index=False,name=None))
    choices=[]
    for name in ('ridge','forest'):
        predictions=pd.read_csv(a.prediction/(name+'_out_of_fold_predictions.csv'),dtype={'fips':str}).set_index('fips').reindex(counties.fips)
        for feature in ('mean_prior','county_baseline','sentiment','combined'):
            values=predictions[feature+'_predicted'].fillna(0).to_numpy()
            if not np.isfinite(values[eligible]).all():
                raise ValueError('Missing held-out predictions on eligible cohort')
            choices.append((name+'_'+feature,values))
    cohort_population=float(n[eligible].sum())
    base=Config(prior=.90,stochastic=True,beta_schedule=schedule,
                eligible_counties=tuple(np.flatnonzero(eligible).tolist()),start_date='2020-11-02')
    manifest={'data_kind':'RETROSPECTIVE EMPIRICAL-INPUT COUNTERFACTUAL WITH REAL OUT-OF-FOLD TBCOV PREDICTIONS',
        'policy_dates':['2020-11-02','2020-11-22'],'days':21,'seeds':a.seeds,'config':asdict(base),
        'network_counties':len(n),'eligible_counties':int(eligible.sum()),'cohort_population':cohort_population,
        'invalid_initialization_counties_excluded':invalid,'source_audit':source_audit,
        'outside_cohort':'No interventions allowed; unobserved responsiveness placeholders are not empirical targets and never affect hazard',
        'prior':'Explicit 0.90 constant CTIS-era prior; fold-trained means also compared separately',
        'prediction_validation':'State-blocked out-of-fold predictions, September sentiment -> October measured proxy; one-month offline validation, not prospective availability validation',
        'rates':'Same retrospectively fitted weekly rates for all policies; no causal spending efficacy claim',
        'budget_fractions_of_eligible_population':[.005,.02]}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    for fraction in (.005,.02):
        cfg=Config(**{**asdict(base),'budget':cohort_population*fraction})
        model=Model(n,response,np.ones(len(n)),mobility,cfg)
        for seed in range(a.seeds):
            for name,estimate in [('none',None),('epidemic',None),('oracle',None)]+choices:
                m,_,_,action=run(model,initial,'predicted' if estimate is not None else name,21,seed+9000,estimate,record=False)
                if action[~eligible].sum()!=0:
                    raise RuntimeError('Intervention outside validated cohort')
                records.append({'fraction':fraction,'seed':seed,'policy':name,'infections':m['infections'],'spent':m['spent']})
            pd.DataFrame(records).to_csv(a.output/'trials.csv',index=False)
            print(f'TBCOV downstream fraction={fraction}, paired seed={seed+1}/{a.seeds}',flush=True)
    summaries=[]
    for fraction,group in pd.DataFrame(records).groupby('fraction'):
        wide=group.pivot(index='seed',columns='policy',values='infections')
        oracle_gain=(wide.epidemic-wide.oracle).mean()
        for policy in wide.columns:
            delta=wide.epidemic-wide[policy]
            half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
            summaries.append({'fraction':fraction,'policy':policy,'mean_infections':wide[policy].mean(),
                'gain_vs_epidemic':delta.mean(),'ci_low':delta.mean()-half,'ci_high':delta.mean()+half,
                'oracle_recovery_fraction':delta.mean()/oracle_gain if oracle_gain>0 else np.nan})
    pd.DataFrame(summaries).to_csv(a.output/'summary.csv',index=False)


if __name__=='__main__':
    main()
