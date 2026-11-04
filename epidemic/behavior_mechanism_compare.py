"""Matched-effect comparison: direct suppression versus persistent adoption.

Targets deliberately specify synthetic intervention strength. They are not
estimated real-world effects. Twenty random response maps are never used to
choose the dose coefficient: matching uses p=.5 and proportional allocation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
from scipy.stats import t


def simulate(initial,matrices,rates,p,guess,mechanism,kappa,policy,record=False):
    n=initial['n'];s=initial['s'].copy();i=initial['i'].copy();r=initial['r'].copy()
    adoption=np.zeros(len(n));belief=np.zeros(len(n));total=0.;records=[]
    observed=p if policy=='oracle' else guess if policy=='predicted' else np.full(len(n),np.average(p,weights=n))
    budget=.1*n.sum()
    for week in range(8):
        local,remote=rates[week];matrix=matrices[week]
        hazard=local*i/n+remote*(matrix@(i/n))
        if policy=='none':allocation=np.zeros(len(n))
        elif policy=='proportional':allocation=.1*n
        else:
            if mechanism=='direct':
                decay=kappa*observed/n
                slope=s*hazard*decay
            else:
                decay=np.full(len(n),kappa)/n
                slope=s*hazard*.8*np.maximum(observed-belief,0)*decay
            low,high=-80.,float(np.log(max(slope.max(),1e-30)))
            for _ in range(55):
                lag=(low+high)/2
                allocation=np.where(decay>0,np.maximum(0,np.log(np.maximum(slope,1e-30))-lag)/np.maximum(decay,1e-30),0)
                if allocation.sum()>budget:low=lag
                else:high=lag
            allocation*=budget/max(allocation.sum(),1e-30)
        if mechanism=='direct':multiplier=np.exp(-kappa*p*allocation/n)
        else:
            reached=-np.expm1(-kappa*allocation/n)
            adoption+=(p-adoption)*reached
            belief+=(observed-belief)*reached
            multiplier=1-.8*adoption
        weekly=0.
        for day in range(7):
            h=(local*i/n+remote*(matrix@(i/n)))*multiplier
            new=s*(-np.expm1(-h));recovered=i*(-np.expm1(-1/7))
            s-=new;i+=new-recovered;r+=recovered
            weekly+=new.sum()
        total+=weekly
        if record:records.append(dict(week=week+1,incidence=weekly,adoption=float(np.average(adoption,weights=n))))
    if not np.allclose(s+i+r,n) or (adoption<0).any() or (adoption>1+1e-12).any():
        raise ValueError('Population or adoption bounds violated')
    return total,records


def main():
    folder=Path('outputs/vaccine_experiment/real')
    out=Path('outputs/behavior_mechanism_compare');out.mkdir(exist_ok=True)
    initial=dict(np.load(folder/'initial.npz'));n=initial['n']
    matrices=[sparse.load_npz(folder/f'mobility_{w}.npz') for w in range(8)]
    rates=pd.read_csv(folder/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    half=np.full(len(n),.5)
    no_intervention=simulate(initial,matrices,rates,half,half,'direct',1,'none')[0]
    # No intervention must agree across mechanisms; frozen p=0 must do nothing.
    assert np.isclose(no_intervention,simulate(initial,matrices,rates,half,half,'adoption',1,'none')[0])
    for mechanism in ['direct','adoption']:
        assert np.isclose(no_intervention,simulate(initial,matrices,rates,np.zeros(len(n)),half,mechanism,10,'proportional')[0])
    calibrations=[];rows=[]
    for target in [.25,.50,.65]:
        for mechanism in ['direct','adoption']:
            def objective(logk):
                value=simulate(initial,matrices,rates,half,half,mechanism,np.exp(logk),'proportional')[0]
                return (no_intervention-value)/no_intervention-target
            if objective(10)<0:
                calibrations.append(dict(target=target,mechanism=mechanism,kappa=np.nan,status='Target exceeds maximum reference effect'))
                continue
            kappa=np.exp(brentq(objective,-12,10,xtol=1e-9))
            calibrations.append(dict(target=target,mechanism=mechanism,kappa=kappa,status='Matched on constant p=.5, population-proportional spending'))
            for draw in range(20):
                rng=np.random.default_rng(771000+draw);p=rng.uniform(0,1,len(n));guess=rng.uniform(0,1,len(n))
                for policy in ['aggregate','oracle','predicted']:
                    value,trace=simulate(initial,matrices,rates,p,guess,mechanism,kappa,policy,record=draw==0)
                    rows.append(dict(target=target,mechanism=mechanism,kappa=kappa,draw=draw,policy=policy,infections=value))
                    if draw==0:pd.DataFrame(trace).to_csv(out/f'trace_{target}_{mechanism}_{policy}.csv',index=False)
            pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
            pd.DataFrame(calibrations).to_csv(out/'reference_calibration.csv',index=False)
            print('Matched effect complete',target,mechanism,'kappa',kappa,flush=True)
    summary=[]
    for (target,mechanism),group in pd.DataFrame(rows).groupby(['target','mechanism']):
        wide=group.pivot(index='draw',columns='policy',values='infections')
        for policy in wide:
            delta=wide['aggregate']-wide[policy];halfwidth=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            summary.append(dict(target=target,mechanism=mechanism,policy=policy,infections=wide[policy].mean(),
                prevention_percent=100*(no_intervention-wide[policy].mean())/no_intervention,
                gain=delta.mean(),gain_percent=100*delta.mean()/wide['aggregate'].mean(),ci_low=delta.mean()-halfwidth,ci_high=delta.mean()+halfwidth))
    result=pd.DataFrame(summary);result.to_csv(out/'summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(kind='Controlled synthetic mechanism comparison on historical SIR network; not calibrated real effects',
        reference='Match 25%, 50%, 65% infection prevention with constant responsiveness .5 and population-proportional spending; independent of test response draws',
        direct='lambda=lambda0*exp(-kappa*p*a/N), effect resets weekly',
        adoption='C_next=C+(p-C)*(1-exp(-kappa*a/N)); lambda=lambda0*(1-.8*C_next); adoption persists without decay for eight weeks',
        interpretation='p is fraction willing to adopt if reached; .8 is assumed maximum transmission reduction at full adoption',
        information='All policies observe epidemic state; baseline knows true population-weighted mean p; adoption state estimated using own p information, not leaked from truth',
        allocation='Analytic greedy minimization of current S*hazard*multiplier; no full-horizon optimization claim',
        cohort='All network counties eligible, same per-capita weekly budget .1 and total budget in both mechanisms',
        uncertainty='Twenty independent Uniform(0,1) county response maps; expected SIR; intervals across maps only',
        no_intervention=no_intervention),indent=2))
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':main()
