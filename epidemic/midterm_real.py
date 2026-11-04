"""Proposal-form counterfactual allocation with measured and predicted proxies.

Greedy is a transparent comparator; transferred PPO is explicitly labeled.
Both raw adherence and remaining-unmasked headroom mappings are retained.
"""
import argparse
from pathlib import Path
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t
import torch
from .proposal_ppo import World, ActorCritic, execute


@lru_cache(maxsize=1)
def inputs():
    folder=Path('outputs/midterm/real')
    initial=np.load(folder/'initial.npz'); behavior=np.load(folder/'behavior.npz')
    rates=pd.read_csv(folder/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    mobility=[sparse.load_npz(folder/f'mobility_{week}.npz') for week in range(8)]
    return initial,behavior,rates,mobility


def load_world(mapping, fraction, estimate='twitter'):
    initial,behavior,rates,mobility=inputs()
    truth=behavior['truth'].copy(); prediction=behavior[estimate].copy()
    prior=.87  # Prespecified CTIS-era training prior; not an evaluation-target mean.
    if mapping=='headroom':
        truth=1-truth; prediction=1-prediction; prior=1-prior
    eligible=behavior['eligible']
    return World(initial['n'],initial['s'],initial['i'],initial['r'],truth,prediction,
                 mobility,rates,fraction,prior,eligible,weeks=8)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=0,help='0 expected-value screen; positive paired stochastic seeds')
    parser.add_argument('--ppo',type=Path,help='PPO model folder, if including transferred policies')
    parser.add_argument('--output',type=Path,default=Path('outputs/midterm/real_screen'))
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    records=[]
    for mapping in ['adherence','headroom']:
        for fraction in [.02,.10]:
            world=load_world(mapping,fraction)
            choices=[('greedy_blind','blind','greedy',None),('greedy_oracle','oracle','greedy',None)]
            for proxy in ['twitter','persistence','history','combined']:
                choices.append(('greedy_'+proxy,'predicted','greedy',proxy))
            choices += [('high_prevalence','blind','high_prevalence',None),('low_prevalence','blind','low_prevalence',None)]
            if args.ppo:
                for information in ['blind','oracle','predicted']:
                    policy=ActorCritic()
                    policy.load_state_dict(torch.load(args.ppo/f'{information}_0.pt',weights_only=True))
                    choices.append(('transferred_ppo_'+information,information,policy,None))
            for seed in range(max(1,args.seeds)):
                for name,information,policy,proxy in choices:
                    current=load_world(mapping,fraction,proxy) if proxy else world
                    outcome,snapshots=execute(current,information,policy,seed=960000+seed,
                        stochastic=args.seeds>0,record=seed==0)
                    records.append({'mapping':mapping,'fraction':fraction,'seed':seed,'policy':name,'infections':outcome})
                    if seed==0:
                        np.savez_compressed(args.output/f'{mapping}_{fraction}_{name}_snapshots.npz',
                            incidence=np.asarray([row['incidence'] for row in snapshots]),
                            allocation=np.asarray([row['allocation'] for row in snapshots]))
                pd.DataFrame(records).to_csv(args.output/'trials.csv',index=False)
                print(f'National proposal replay {mapping}, budget{fraction}, seed {seed+1}/{max(1,args.seeds)}',flush=True)
    summaries=[]
    for (mapping,fraction),group in pd.DataFrame(records).groupby(['mapping','fraction']):
        wide=group.pivot(index='seed',columns='policy',values='infections')
        for name in wide:
            difference=wide.greedy_blind-wide[name]
            half=t.ppf(.975,len(difference)-1)*difference.std(ddof=1)/np.sqrt(len(difference)) if len(difference)>1 else np.nan
            summaries.append({'mapping':mapping,'fraction':fraction,'policy':name,'infections':wide[name].mean(),
                'gain_vs_blind':difference.mean(),'ci_low':difference.mean()-half,'ci_high':difference.mean()+half,
                'relative_percent':100*difference.mean()/wide.greedy_blind.mean()})
    pd.DataFrame(summaries).to_csv(args.output/'summary.csv',index=False)
    (args.output/'manifest.json').write_text(json.dumps({'data_kind':'PROPOSAL-FORM EMPIRICAL-INPUT COUNTERFACTUAL',
        'mappings':{'adherence':'p=measured mask fraction; proposal primary assumption',
                    'headroom':'p=1-measured mask fraction; explicit alternative capacity-for-change assumption, not a fitted causal response'},
        'dates':['2020-11-02','2020-12-27'],'weekly_budget_fractions':[.02,.10],
        'prior':{'adherence':.87,'headroom':.13},'seeds':args.seeds,
        'confidence_intervals':'conditional paired epidemic noise only; not causal, parameter or proxy uncertainty',
        'ppo':str(args.ppo) if args.ppo else None,
        'ppo_limitation':'If present, PPO trained on independent synthetic networks and transferred without real-data fine-tuning; do not describe as nationally trained PPO',
        'source_manifest':'outputs/midterm/real/manifest.json'},indent=2))


if __name__=='__main__':
    main()
