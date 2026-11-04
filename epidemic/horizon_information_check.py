"""Full-horizon resource planning diagnostic with a checked exact adjoint.

Both information levels choose fixed eight-week schedules; this is neither
PPO nor an optimal adaptive-policy claim. Unknown p uses its exact global mean.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import t


def objective(z,n,s0,i0,matrices,rates,p,kappa,gradient=True):
    weeks=len(matrices);z=np.asarray(z).reshape(weeks,len(n))
    shares=softmax(np.log(n)[None,:]+z,axis=1);allocation=.1*n.sum()*shares
    s=s0.copy();i=i0.copy();history=[];total=0.;rho=-np.expm1(-1/7)
    for week in range(weeks):
        g=np.exp(-kappa*p*allocation[week]/n);local,remote=rates[week];matrix=matrices[week]
        for _ in range(7):
            h=local*i/n+remote*(matrix@(i/n));survival=np.exp(-g*h);q=1-survival
            if gradient:history.append((week,s.copy(),survival,q,h,g))
            new=s*q;total+=new.sum();s-=new;i+=(new-rho*i)
    if not gradient:return total
    adj_s=np.zeros(len(n));adj_i=np.zeros(len(n));grad_a=np.zeros_like(allocation)
    for week,s,survival,q,h,g in reversed(history):
        m=1-adj_s+adj_i;d=m*s*survival*g
        grad_a[week]+=-m*s*survival*h*kappa*p*g/n
        local,remote=rates[week]
        new_i=(1-rho)*adj_i+(local*d+remote*(matrices[week].T@d))/n
        adj_s=adj_s+m*q;adj_i=new_i
    grad_z=allocation*(grad_a-(shares*grad_a).sum(axis=1)[:,None])
    return total/1e6,grad_z.ravel()/1e6


def main():
    out=Path('outputs/horizon_information_check');out.mkdir(exist_ok=True)
    # Independent finite-difference validation of the multistep state adjoint.
    rng=np.random.default_rng(42001);n=np.array([1000.,1500.,2000.]);s=n*.97;i=n*.01
    w=sparse.csr_matrix([[0,.6,.4],[.4,0,.6],[.7,.3,0]])
    z=rng.normal(0,.2,6);p=np.array([.1,.5,.9]);matrices=[w,w];rates=np.tile([.2,.04],(2,1))
    value,grad=objective(z,n,s,i,matrices,rates,p,3.)
    numerical=[]
    for coordinate in range(len(z)):
        plus=z.copy();minus=z.copy();plus[coordinate]+=1e-5;minus[coordinate]-=1e-5
        numerical.append((objective(plus,n,s,i,matrices,rates,p,3.,False)-objective(minus,n,s,i,matrices,rates,p,3.,False))/2e-5/1e6)
    np.testing.assert_allclose(grad,numerical,rtol=2e-5,atol=1e-10)
    print('Full-horizon adjoint finite-difference check passed',flush=True)
    folder=Path('outputs/vaccine_experiment/real');initial=np.load(folder/'initial.npz');n=initial['n']
    matrices=[sparse.load_npz(folder/f'mobility_{week}.npz') for week in range(8)]
    rates=pd.read_csv(folder/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    calibration=pd.read_csv('outputs/behavior_mechanism_compare/reference_calibration.csv')
    kappa=float(calibration[(calibration.target==.5)&(calibration.mechanism=='direct')].kappa.iloc[0])
    records=[];diagnostics=[]
    for draw in range(20):
        p=np.random.default_rng(771000+draw).uniform(0,1,len(n));mean=np.full(len(n),np.average(p,weights=n))
        initial_z=np.zeros(8*len(n));blind_z=None
        for policy,estimate in [('aggregate',mean),('oracle',p)]:
            start=initial_z if blind_z is None else blind_z
            args=(n,initial['s'],initial['i'],matrices,rates,estimate,kappa)
            fit=minimize(objective,start,args=args,jac=True,method='L-BFGS-B',bounds=[(-12,12)]*len(start),
                options={'maxiter':60,'ftol':1e-10,'gtol':1e-7,'maxls':20,'maxcor':8})
            # Keep feasible starting plan if line search does not improve it.
            chosen=fit.x if fit.fun<=objective(start,*args)[0] else start
            if policy=='aggregate':blind_z=chosen
            value=objective(chosen,n,initial['s'],initial['i'],matrices,rates,p,kappa,False)
            records.append(dict(draw=draw,policy=policy,infections=value))
            diagnostics.append(dict(draw=draw,policy=policy,iterations=fit.nit,evaluations=fit.nfev,success=bool(fit.success),message=str(fit.message)))
            if draw==0:np.savez_compressed(out/f'{policy}_schedule.npz',z=chosen,p=p)
            pd.DataFrame(records).to_csv(out/'trials.csv',index=False)
            pd.DataFrame(diagnostics).to_csv(out/'optimization.csv',index=False)
            print('Horizon planning draw',draw+1,policy,'iterations',fit.nit,'infections',round(value),flush=True)
    wide=pd.DataFrame(records).pivot(index='draw',columns='policy',values='infections')
    delta=wide['aggregate']-wide['oracle'];half=t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
    result=dict(aggregate_infections=wide['aggregate'].mean(),oracle_infections=wide['oracle'].mean(),gain=delta.mean(),gain_percent=100*delta.mean()/wide['aggregate'].mean(),ci_low=delta.mean()-half,ci_high=delta.mean()+half)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(kind='Synthetic responsiveness, empirical-input deterministic epidemic; full-horizon fixed schedules',
        mechanism='Original direct exponential suppression, dose matched to 50% reference prevention',kappa=kappa,
        optimizer='L-BFGS-B, exact manually differentiated daily SIR adjoint; 60-iteration fixed cap, reported termination diagnostics',
        information='Baseline plans using actual global mean p; oracle initializes at same feasible baseline schedule and optimizes with true county p',
        limitation='Local numerical optimization, not proven optimal; open-loop schedules and no policy feedback; not causal or empirical p validation',
        uncertainty='20 random Uniform(0,1) response maps; confidence intervals across maps'),indent=2))
    print(result,flush=True)


if __name__=='__main__':main()
