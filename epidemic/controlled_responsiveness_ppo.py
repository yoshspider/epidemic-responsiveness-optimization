"""PPO mechanism validation with learnable Uniform(0,1) synthetic behavior.

Historical epidemic network/initial state; synthetic behavior is explicitly
separate from the empirical proxy experiment. No empirical forecasting claim.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import torch
from .proposal_ppo import World,ActorCritic,train,execute


def response(count,seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(count,6))
    p=ndtr((1.2*x[:,0]-.9*x[:,1]+.6*x[:,2])/np.sqrt(2.61))
    return x,p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=8192);args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path(f'outputs/controlled_responsiveness_ppo/seed{args.seed}');out.mkdir(parents=True,exist_ok=True)
    source=Path('outputs/vaccine_experiment/real')
    initial=np.load(source/'initial.npz');n=initial['n']
    matrices=[sparse.load_npz(source/f'mobility_{w}.npz') for w in range(8)]
    rates=pd.read_csv(source/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    x,p=response(6000,880001)
    labels=np.clip(p+np.random.default_rng(880002).normal(0,.05,len(p)),0,1)
    learner=RandomForestRegressor(n_estimators=150,min_samples_leaf=5,n_jobs=1,random_state=42).fit(x,labels)
    xx,pp=response(3000,880003);phat=learner.predict(xx)
    def factory(seed,evaluation=False):
        rng=np.random.default_rng(seed)
        x,p=response(len(n),seed+1000000);estimate=learner.predict(x)
        infected=initial['i'].copy() if evaluation else np.minimum(np.rint(initial['i']*rng.uniform(.7,1.3)),n-initial['r'])
        beta=rng.uniform(.13,.27)
        return World(n,n-infected-initial['r'],infected,initial['r'],np.tile(p,(8,1)),np.tile(estimate,(8,1)),
            matrices if evaluation else matrices[0],rates if evaluation else np.tile([.85*beta,.15*beta],(8,1)),
            .1,.5,np.ones(len(n),bool),weeks=8,effectiveness=3.)
    (out/'manifest.json').write_text(json.dumps(dict(kind='Synthetic learnable Uniform county responsiveness on historical epidemic network',
        predictor_r2=r2_score(pp,phat),predictor_mae=mean_absolute_error(pp,phat),steps_per_policy=args.steps,training_seed=args.seed,
        behavior='Normal-CDF linear latent features gives Uniform(0,1); synthetic measurement noise SD .05; forest trained on independent simulated counties',
        training='Random initial incidence .7–1.3 and total beta .13–.27; first mobility matrix only; generated behavior worlds seeds below 500000',
        evaluation='Fixed March–April historical epidemic replay with independent generated behavior seeds 992000–992019; expected dynamics',
        effect='kappa=3, weekly per-capita budget .1, all network counties eligible',
        warmstart='Matched original 24576-decision synthetic checkpoint for each information level and seed',
        limitation='Neither synthetic predictor features nor responsiveness are observed empirical measurements; no empirical forecast claim'),indent=2))
    for name,info in [('blind','aggregate'),('oracle','oracle'),('predicted','predicted')]:
        path=out/f'{name}.pt';marker=out/f'{name}.complete.json'
        if not marker.exists():
            train(info,learner,.5,args.seed,args.steps,path,world_factory=factory,
                initial_checkpoint=Path('outputs/proposal_ppo_confirmation')/f'{name}_{args.seed}.pt')
            marker.write_text(json.dumps(dict(complete=True,steps=args.steps)))
    records=[]
    for draw in range(20):
        world=factory(992000+draw,evaluation=True)
        for name,info in [('blind','aggregate'),('oracle','oracle'),('predicted','predicted')]:
            policy=ActorCritic();policy.load_state_dict(torch.load(out/f'{name}.pt',weights_only=True))
            total,_=execute(world,info,policy,stochastic=False)
            records.append(dict(training_seed=args.seed,draw=draw,policy='ppo_'+name,infections=total))
            if args.seed==0:
                total,_=execute(world,info,'greedy',stochastic=False)
                records.append(dict(training_seed=-1,draw=draw,policy='greedy_'+name,infections=total))
        pd.DataFrame(records).to_csv(out/'trials.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(complete=True)))
    print('Controlled PPO seed complete',args.seed,flush=True)


if __name__=='__main__':main()
