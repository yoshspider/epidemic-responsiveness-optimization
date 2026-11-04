"""Diagnostic: independent Uniform(0,1) county responsiveness, unchanged SIR."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
from .proposal_ppo import World,execute


def main():
    folder=Path('outputs/vaccine_experiment/real')
    out=Path('outputs/random_responsiveness');out.mkdir(exist_ok=True)
    initial=np.load(folder/'initial.npz');n=initial['n']
    counties=pd.read_csv(folder/'counties.csv',dtype={'fips':str})
    cohort=pd.read_csv('outputs/vaccine_experiment/eligible_counties.csv',dtype={'fips':str})
    eligible=counties.fips.isin(cohort.fips).to_numpy()
    mobility=[sparse.load_npz(folder/f'mobility_{w}.npz') for w in range(8)]
    rates=pd.read_csv(folder/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    rows=[]
    for draw in range(20):
        rng=np.random.default_rng(771000+draw)
        p=np.tile(rng.uniform(0,1,len(n)),(8,1))
        guess=np.tile(rng.uniform(0,1,len(n)),(8,1))
        for fraction in [.02,.1]:
            world=World(n,initial['s'],initial['i'],initial['r'],p,guess,mobility,rates,fraction,.5,eligible,weeks=8)
            for info in ['aggregate','oracle','predicted']:
                total,_=execute(world,info,'greedy',stochastic=False)
                rows.append(dict(draw=draw,fraction=fraction,policy=info,infections=total))
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
        print('Random responsiveness draw',draw+1,flush=True)
    summary=[]
    for fraction,group in pd.DataFrame(rows).groupby('fraction'):
        wide=group.pivot(index='draw',columns='policy',values='infections')
        for policy in wide:
            delta=wide['aggregate']-wide[policy]
            half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            summary.append(dict(fraction=fraction,policy=policy,infections=wide[policy].mean(),gain=delta.mean(),percent=100*delta.mean()/wide['aggregate'].mean(),ci_low=delta.mean()-half,ci_high=delta.mean()+half))
    result=pd.DataFrame(summary);result.to_csv(out/'summary.csv',index=False)
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()
