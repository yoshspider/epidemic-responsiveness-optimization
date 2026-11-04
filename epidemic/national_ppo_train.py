"""Fine-tune identical PPO policies on the full historical county network.

Training response labels are October only; held-out November/December labels
never enter the factory. Epidemic training states are randomized around the
known initial state. Current evaluation rates and later mobility are not used
for training: one first-week matrix and randomized transmission are used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .midterm_real import inputs
from .proposal_ppo import World, train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=8192)
    args=parser.parse_args()
    torch.set_num_threads(1)
    folder=Path(f'outputs/national_ppo_seed{args.seed}'); folder.mkdir(exist_ok=True)
    initial,behavior,_,matrices=inputs()
    counties=pd.read_csv('outputs/midterm/real/counties.csv',dtype={'fips':str})
    masks=pd.read_csv('outputs/multi_month/mask_county_month.csv',dtype={'fips':str})
    masks=masks[masks.signal=='smoothed_wwearing_mask']
    october=masks[masks.month=='2020-10'].set_index('fips').responsiveness.reindex(counties.fips)
    september=masks[masks.month=='2020-09'].set_index('fips').responsiveness.reindex(counties.fips)
    eligible=behavior['eligible']; n=initial['n']
    if october.iloc[np.flatnonzero(eligible)].isna().any():raise ValueError('Missing October training proxy')
    prior=float(october[eligible].mean())
    past=pd.read_csv('outputs/tbcov_prediction/forest_out_of_fold_predictions.csv',dtype={'fips':str}).set_index('fips').reindex(counties.fips)
    twitter=past.sentiment_predicted.fillna(prior).to_numpy()
    survey=september.fillna(prior).to_numpy()
    truth=october.fillna(0).to_numpy()
    metadata={'training_seed':args.seed,'fine_tuning_steps_per_policy':args.steps,
        'synthetic_warmstart_steps_per_policy':24576,'training_proxy_month':'2020-10',
        'forecast_proxy_training':'September survey for survey policy; September tweets state-held-out October predictions for Twitter policy',
        'truth':'Observed October proxy plus independently simulated SD .02 response uncertainty; clipped to [0,1], not fabricated observed labels',
        'epidemic_training':'known November 2 initial state times simulated incidence multiplier .7–1.3, randomized local+mobility rate .13–.27; fixed 15% mobility share',
        'mobility_training':'first observed week only, held fixed within each training world; later evaluation weekly networks excluded',
        'network_counties':len(n),'eligible_counties':int(eligible.sum()),'training_prior':prior,
        'evaluation':'November–December proxy labels and retrospectively fitted evaluation transmission rates are excluded from this training factory',
        'architecture':'same 22->64->10 Gaussian score actor and 22->64->1 critic, softmax weekly spending, for all four policies',
        'interpretation':'Retrospective simulation/domain adaptation, not a validated as-of real deployment'}
    (folder/'manifest.json').write_text(json.dumps(metadata,indent=2))
    for name,information,estimated in [('blind','blind',twitter),('oracle','oracle',twitter),
                                        ('twitter','predicted',twitter),('survey','predicted',survey)]:
        destination=folder/f'{name}.pt'
        if destination.exists() and (folder/f'{name}.complete.json').exists():continue
        def factory(seed):
            rng=np.random.default_rng(seed)
            infected=np.maximum(0,np.rint(initial['i']*rng.uniform(.7,1.3)))
            removed=initial['r'].copy()
            infected=np.minimum(infected,n-removed)
            response=truth.copy()
            response[eligible]=np.clip(response[eligible]+rng.normal(0,.02,eligible.sum()),0,1)
            beta=rng.uniform(.13,.27)
            return World(n,n-infected-removed,infected,removed,np.tile(response,(8,1)),
                np.tile(estimated,(8,1)),matrices[0],np.tile([.85*beta,.15*beta],(8,1)),
                float(rng.choice([.02,.10])),prior,eligible,weeks=8)
        warm='predicted' if name in ['twitter','survey'] else name
        source=Path('outputs/proposal_ppo_confirmation')/f'{warm}_{args.seed}.pt'
        train(information,None,prior,args.seed,args.steps,destination,world_factory=factory,initial_checkpoint=source)
        (folder/f'{name}.complete.json').write_text(json.dumps({'complete':True,'steps':args.steps,'warmstart':str(source)}))
    (folder/'complete.json').write_text(json.dumps({'complete':True,'seed':args.seed,'steps_per_policy':args.steps}))


if __name__=='__main__':main()
