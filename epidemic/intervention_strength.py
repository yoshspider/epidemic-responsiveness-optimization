"""Prespecified intervention-strength sensitivity, not effect calibration."""
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
from .proposal_ppo import World,execute


def main():
    suite=unittest.defaultTestLoader.discover('tests',pattern='test_proposal.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise RuntimeError('Intervention checks failed')
    folder=Path('outputs/vaccine_experiment/real')
    out=Path('outputs/intervention_strength');out.mkdir(exist_ok=True)
    initial=np.load(folder/'initial.npz');n=initial['n']
    counties=pd.read_csv(folder/'counties.csv',dtype={'fips':str})
    cohort=pd.read_csv('outputs/vaccine_experiment/eligible_counties.csv',dtype={'fips':str})
    original=counties.fips.isin(cohort.fips).to_numpy()
    mobility=[sparse.load_npz(folder/f'mobility_{w}.npz') for w in range(8)]
    rates=pd.read_csv(folder/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    rows=[]
    def evaluate(label,draw,p,estimate,eligible,scale):
        for info in ['none','aggregate','oracle','predicted']:
            world=World(n,initial['s'],initial['i'],initial['r'],p,estimate,mobility,rates,
                0. if info=='none' else .1,.5,eligible,weeks=8,effectiveness=scale)
            total,_=execute(world,'aggregate' if info=='none' else info,'greedy',stochastic=False)
            rows.append(dict(scenario=label,draw=draw,effectiveness=scale,policy=info,infections=total))
    for draw in range(20):
        rng=np.random.default_rng(771000+draw)
        p=np.tile(rng.uniform(0,1,len(n)),(8,1));guess=np.tile(rng.uniform(0,1,len(n)),(8,1))
        for label,eligible in [('random_common71',original),('random_all_network',np.ones(len(n),bool))]:
            for scale in [1.,3.,10.]:evaluate(label,draw,p,guess,eligible,scale)
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
        print('Strength sensitivity random assignment',draw+1,flush=True)
    # Same measured proxies and previous-month survey forecasts as prior study.
    dates=pd.date_range('2021-03-01',periods=8,freq='7D')
    for metric in ['acceptance','recommendation']:
        tables={}
        for month in ['2021-02','2021-03','2021-04']:
            data=pd.read_csv(f'data/raw/vaccine_experiment/{metric}_{month}.csv',dtype={'geo_value':str})
            data=data[(data.missing_value==0)&(~data.geo_value.str.endswith('000'))]
            tables[month]=(data.groupby('geo_value').value.mean()/100).reindex(counties.fips).fillna(0).to_numpy()
        p=np.asarray([tables[d.strftime('%Y-%m')] for d in dates])
        past=np.asarray([tables[(d.to_period('M')-1).strftime('%Y-%m')] for d in dates])
        for scale in [1.,3.,10.]:evaluate(metric,0,p,past,original,scale)
    frame=pd.DataFrame(rows);frame.to_csv(out/'trials.csv',index=False)
    summary=[]
    for (scenario,scale),group in frame.groupby(['scenario','effectiveness']):
        wide=group.pivot(index='draw',columns='policy',values='infections')
        for policy in wide:
            delta=wide['aggregate']-wide[policy]
            half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
            prevented=wide['none']-wide[policy]
            summary.append(dict(scenario=scenario,effectiveness=scale,policy=policy,infections=wide[policy].mean(),
                prevented_vs_none=prevented.mean(),prevention_percent=100*prevented.mean()/wide['none'].mean(),
                gain_vs_aggregate=delta.mean(),gain_percent=100*delta.mean()/wide['aggregate'].mean(),ci_low=delta.mean()-half,ci_high=delta.mean()+half))
    result=pd.DataFrame(summary);result.to_csv(out/'summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(kind='Sensitivity scenarios, NOT calibrated intervention effects',equation='lambda = exp(-kappa*p*a/N)*lambda0',kappa=[1,3,10],budget=.1,
        interpretation='kappa converts population-normalized effort into transmission reduction; original proposal kappa=1',
        random='20 independent county Uniform(0,1) assignments fixed over time; uninformative random forecast control',
        empirical='Same 71 survey-covered eligible counties and past-month forecasts as prior experiment',
        coverage='All-network random scenario expands eligibility AND total budget to 10% of that population; not a fixed-total-budget comparison across cohorts',
        allocation='Exact instantaneous water filling, not full-horizon optimal or newly trained PPO',
        uncertainty='Expected SIR; intervals across random responsiveness assignments only'),indent=2))
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()
