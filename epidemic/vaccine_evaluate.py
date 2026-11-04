"""Matched-cohort vaccine-attitude proxy allocation sensitivity; not causal evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
from .proposal_ppo import World,execute


def main():
    out=Path('outputs/vaccine_experiment');out.mkdir(exist_ok=True)
    folder=out/'real';initial=np.load(folder/'initial.npz')
    counties=pd.read_csv(folder/'counties.csv',dtype={'fips':str})
    n=initial['n'];surveys={};eligible=np.ones(len(n),bool)
    for metric in ['acceptance','recommendation']:
        tables={}
        for month in ['2021-02','2021-03','2021-04']:
            data=pd.read_csv(f'data/raw/vaccine_experiment/{metric}_{month}.csv',dtype={'geo_value':str})
            data=data[(data.missing_value==0)&(~data.geo_value.str.endswith('000'))]
            grouped=data.groupby('geo_value').agg(value=('value','mean'),days=('value','size'))
            tables[month]=(grouped.value.where(grouped.days>=20)/100).reindex(counties.fips).to_numpy()
            eligible &= np.isfinite(tables[month])
        surveys[metric]=tables
    rates=pd.read_csv(folder/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    mobility=[sparse.load_npz(folder/f'mobility_{week}.npz') for week in range(8)]
    dates=pd.date_range('2021-03-01',periods=8,freq='7D')
    records=[];stats=[]
    for metric,tables in surveys.items():
        truth=np.asarray([np.nan_to_num(tables[d.strftime('%Y-%m')]) for d in dates])
        past=np.asarray([np.nan_to_num(tables[(d.to_period('M')-1).strftime('%Y-%m')]) for d in dates])
        prior=float(np.average(tables['2021-02'][eligible],weights=n[eligible]))
        for month in ['2021-03','2021-04']:
            p=tables[month][eligible];w=n[eligible];mean=np.average(p,weights=w)
            stats.append(dict(metric=metric,month=month,counties=int(eligible.sum()),population=int(w.sum()),mean=mean,sd=np.sqrt(np.average((p-mean)**2,weights=w))))
        for fraction in [0.,.02,.10]:
            world=World(n,initial['s'],initial['i'],initial['r'],truth,past,mobility,rates,fraction,prior,eligible,weeks=8)
            for seed in [-1,0,1,2,3,4,5,6,7]:
                for info in (['blind'] if fraction==0 else ['blind','aggregate','oracle','predicted']):
                    total,_=execute(world,info,'greedy',seed=991000+max(0,seed),stochastic=seed>=0)
                    records.append(dict(metric=metric,fraction=fraction,seed=seed,policy=info,infections=total))
                pd.DataFrame(records).to_csv(out/'trials.csv',index=False)
                print(metric,fraction,seed,flush=True)
    summary=[]
    frame=pd.DataFrame(records)
    for (metric,fraction),group in frame.groupby(['metric','fraction']):
        for mode in ['expected','stochastic']:
            wide=group[group.seed.eq(-1) if mode=='expected' else group.seed.ge(0)].pivot(index='seed',columns='policy',values='infections')
            for name in wide:
                base='aggregate' if 'aggregate' in wide else 'blind';delta=wide[base]-wide[name]
                half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
                summary.append(dict(metric=metric,fraction=fraction,mode=mode,policy=name,infections=wide[name].mean(),gain_vs_baseline=delta.mean(),percent=100*delta.mean()/wide[base].mean(),ci_low=delta.mean()-half,ci_high=delta.mean()+half))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(stats).to_csv(out/'variation.csv',index=False)
    counties[eligible].to_csv(out/'eligible_counties.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(kind='Retrospective empirical-input SIR counterfactual; attitudes are assumed spending responsiveness, not causal measurements',dates=['2021-03-01','2021-04-25'],budget='Same population-scaled weekly effort, unchanged exponential intervention',cohort=int(eligible.sum()),population=int(n[eligible].sum()),prediction='Previous-month survey persistence; Twitter forecast not yet evaluated',baseline='aggregate knows true current population-weighted mean; blind knows February mean',uncertainty='Eight paired epidemic seeds only; excludes proxy and parameter uncertainty',confounding='2021 SIR omits explicit vaccination, so fitted transmission absorbs vaccination and other time-varying effects'),indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
