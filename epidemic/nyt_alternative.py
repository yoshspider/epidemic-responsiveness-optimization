"""Independent NYT baseline behavior proxy with nationwide coverage.

This is a stable-trait sensitivity, not a claim that July adherence measures
November causal responsiveness. All policies receive the same aggregate mean.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
import torch
from .midterm_real import inputs
from .proposal_ppo import World,execute


def calibrate_mean(values,weights,target):
    # All comparison policies receive the same population-weighted aggregate;
    # no individual evaluation county labels enter this intercept adjustment.
    shift=brentq(lambda offset: np.average(np.clip(values+offset,0,1),weights=weights)-target,-1,1)
    return np.clip(values+shift,0,1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--political',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path('outputs/nyt_alternative_politics' if args.political else 'outputs/nyt_alternative');out.mkdir(exist_ok=True)
    raw=pd.read_csv('data/raw/nyt_mask_use.csv',dtype={'COUNTYFP':str})
    raw['fips']=raw.COUNTYFP.str.zfill(5)
    raw['truth']=raw.ALWAYS+raw.FREQUENTLY+.5*raw.SOMETIMES
    pop=pd.read_csv('data/transformed/census_population_2019.csv',dtype={'fips':str})
    svi=pd.read_csv('data/raw/alternative_sources/cdc_svi2018_counties.csv',dtype={'FIPS':str}).copy()
    svi['fips']=svi.FIPS.str.zfill(5)
    cols=['E_PCI','EP_POV','EP_UNEMP','EP_NOHSDP','EP_AGE65','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_NOVEH','EP_UNINSUR','AREA_SQMI']
    for col in cols:svi[col]=pd.to_numeric(svi[col],errors='coerce').where(lambda x:x>=0)
    frame=raw[['fips','truth']].merge(pop[['fips','population']],on='fips',validate='one_to_one').merge(svi[['fips']+cols],on='fips',how='left',validate='one_to_one')
    frame['log_population']=np.log(frame.population)
    frame['log_density']=np.log1p(frame.population/frame.AREA_SQMI.replace(0,np.nan))
    sets={'population':['log_population'],'demographics':['log_population','log_density']+cols[:-1]}
    if args.political:
        politics=pd.read_csv('data/raw/alternative_sources/election2016_features.csv',dtype={'fips':str})
        frame=frame.merge(politics,on='fips',how='left',validate='one_to_one')
        sets={'political':['log_population','democratic_share_2016'],
              'demographics_political':['log_population','log_density']+cols[:-1]+['democratic_share_2016']}
    quality=[]
    for model in ['ridge','forest']:
        for name,features in sets.items():
            estimates=np.zeros(len(frame))
            for train,test in GroupKFold(5).split(frame,groups=frame.fips.str[:2]):
                estimator=Ridge(alpha=10) if model=='ridge' else RandomForestRegressor(n_estimators=250,min_samples_leaf=8,random_state=42,n_jobs=2)
                pipeline=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),estimator)
                pipeline.fit(frame.iloc[train][features],frame.iloc[train].truth)
                estimates[test]=np.clip(pipeline.predict(frame.iloc[test][features]),0,1)
            frame[model+'_'+name]=estimates
            quality.append({'model':model,'features':name,'counties':len(frame),'mae':mean_absolute_error(frame.truth,estimates),'r2':r2_score(frame.truth,estimates)})
    frame.to_csv(out/'state_heldout_predictions.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'prediction_metrics.csv',index=False)
    print(pd.DataFrame(quality).to_string(index=False),flush=True)
    initial,_,rates,mobility=inputs()
    counties=pd.read_csv('outputs/midterm/real/counties.csv',dtype={'fips':str})
    joined=frame.set_index('fips').reindex(counties.fips)
    if joined.truth.isna().any():raise ValueError('Unobserved national NYT proxy')
    mean=float(np.average(joined.truth,weights=initial['n']))
    records=[]
    for fraction in [.02,.10]:
        for seed in range(8):
            choices=[('aggregate_baseline','blind',np.full(len(joined),mean)),('oracle','oracle',joined.truth.to_numpy())]
            for model in ['ridge','forest']:
                for name in sets:
                    values=calibrate_mean(joined[model+'_'+name].to_numpy(),initial['n'],mean)
                    choices.append((model+'_'+name,'predicted',values))
            for name,info,estimated in choices:
                world=World(initial['n'],initial['s'],initial['i'],initial['r'],np.tile(joined.truth.to_numpy(),(8,1)),
                    np.tile(estimated,(8,1)),mobility,rates,fraction,mean,None,weeks=8)
                infections,_=execute(world,info,'greedy',980000+seed)
                records.append({'fraction':fraction,'seed':seed,'policy':name,'infections':infections})
            pd.DataFrame(records).to_csv(out/'trials.csv',index=False)
            print(f'NYT nationwide alternative fraction{fraction}, seed {seed+1}/8',flush=True)
    summary=[]
    for fraction,group in pd.DataFrame(records).groupby('fraction'):
        wide=group.pivot(index='seed',columns='policy',values='infections')
        for name in wide:
            delta=wide.aggregate_baseline-wide[name]
            half=t.ppf(.975,7)*delta.std(ddof=1)/np.sqrt(8)
            summary.append({'fraction':fraction,'policy':name,'baseline_infections':wide.aggregate_baseline.mean(),
                'infections':wide[name].mean(),'gain':delta.mean(),'ci_low':delta.mean()-half,'ci_high':delta.mean()+half,
                'relative_percent':100*delta.mean()/wide.aggregate_baseline.mean()})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'data_kind':'INDEPENDENT-SURVEY BASELINE-TRAIT COUNTERFACTUAL SENSITIVITY',
        'proxy':'NYT July 2–14 2020 ALWAYS+FREQUENTLY+0.5*SOMETIMES, assumed stable behavioral trait during November–December replay; not current measured or causal response',
        'why':'NYT is an independent source in the proposal, covers nearly every county, and retains geographic variation before later high masking levels',
        'prediction':'five held-out state folds; pre-pandemic CDC SVI2018 ACS covariates; fixed models; no test tuning',
        'feature_sets':sets,
        'political_source':'2016 community county compilation, statewide Alaska replicas excluded' if args.political else None,
        'macro_information':'Every policy knows the same population-weighted NYT aggregate mean; predictor intercepts adjusted to that aggregate without using individual test labels',
        'network_and_treatment_counties':len(joined),'prior':mean,'dates':['2020-11-02','2020-12-27'],
        'weekly_budget_fractions':[.02,.10],'seeds':8,'allocator':'identical analytic myopic water-filling comparator, not PPO',
        'limitations':'Retrospective epidemic rate fit; stable-trait proxy assumption unvalidated; intervals condition on this model and quantify epidemic noise only'},indent=2))


if __name__=='__main__':main()
