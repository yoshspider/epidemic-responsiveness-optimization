"""Forward-proxy evaluation of full-network-trained PPO; fixed final models."""
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t
import torch
from .proposal_ppo import ActorCritic,execute
from .midterm_real import load_world


def main():
    torch.set_num_threads(1)
    out=Path('outputs/national_ppo_evaluation');out.mkdir(exist_ok=True)
    for _ in range(360):
        if all(Path(f'outputs/national_ppo_seed{seed}/complete.json').exists() for seed in range(3)):break
        print('Waiting for the fixed full-network PPO checkpoints',flush=True);time.sleep(15)
    else:raise TimeoutError('National training did not finish within 90 minutes')
    models={}
    for seed in range(3):
        folder=Path(f'outputs/national_ppo_seed{seed}')
        for name in ['blind','oracle','twitter','survey']:
            model=ActorCritic();model.load_state_dict(torch.load(folder/f'{name}.pt',weights_only=True))
            models[seed,name]=model
    meta=json.loads(Path('outputs/national_ppo_seed0/manifest.json').read_text())
    prior=meta['training_prior']
    rows=[]
    for fraction in [.02,.10]:
        world=replace(load_world('adherence',fraction),prior=prior)
        for epidemic_seed in range(8):
            for training_seed in range(3):
                for name in ['blind','oracle','twitter','survey','oracle_with_aggregate_only']:
                    information='blind' if name=='blind' else 'oracle' if name=='oracle' else 'aggregate' if name=='oracle_with_aggregate_only' else 'predicted'
                    estimate='persistence' if name=='survey' else 'twitter'
                    current=replace(load_world('adherence',fraction,estimate),prior=prior)
                    policy=models[training_seed,'oracle' if name=='oracle_with_aggregate_only' else name]
                    outcome,_=execute(current,information,policy,960000+epidemic_seed)
                    rows.append({'fraction':fraction,'training_seed':training_seed,'epidemic_seed':epidemic_seed,'policy':name,'infections':outcome})
            pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
            print(f'Full-network PPO evaluation budget{fraction}, epidemic seed {epidemic_seed+1}/8',flush=True)
    data=pd.DataFrame(rows);summary=[];seed_summary=[]
    for fraction,group in data.groupby('fraction'):
        wide=group.pivot(index=['training_seed','epidemic_seed'],columns='policy',values='infections')
        for training_seed,seed_group in wide.groupby(level='training_seed'):
            for name in wide:
                gain=seed_group.blind-seed_group[name]
                seed_summary.append({'fraction':fraction,'training_seed':training_seed,'policy':name,'gain_vs_blind':gain.mean()})
        averaged=wide.groupby('epidemic_seed').mean()
        for comparator in ['blind','oracle_with_aggregate_only']:
            for name in averaged:
                gain=averaged[comparator]-averaged[name]
                half=t.ppf(.975,7)*gain.std(ddof=1)/np.sqrt(8)
                summary.append({'fraction':fraction,'policy':name,'comparator':comparator,
                    'baseline_infections':averaged[comparator].mean(),'infections':averaged[name].mean(),
                    'gain':gain.mean(),'ci_low':gain.mean()-half,'ci_high':gain.mean()+half,
                    'relative_percent':100*gain.mean()/averaged[comparator].mean()})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(seed_summary).to_csv(out/'training_seed_effects.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'data_kind':'FULL-NETWORK PPO EMPIRICAL-INPUT COUNTERFACTUAL',
        'network_counties':3122,'behavioral_cohort':381,'training':'October response proxies only, same full network, 8192 fine-tuning weekly decisions per policy/seed after 24576 synthetic decisions; three training seeds',
        'evaluation':'November 2–December 27, true November/December CTIS proxies; actual eight weekly flows and retrospective rates; no evaluation-label fitting of the PPO actor',
        'information_levels':['blind training prior','observed current proxy','Twitter forecast','previous county survey'],
        'secondary_ablation':'oracle_with_aggregate_only uses the SAME learned oracle-policy weights but masks county p to the current population-weighted aggregate',
        'uncertainty':'paired epidemic-noise t intervals after averaging the three trained policies within seed; training-seed effects reported separately; no causal/parameter uncertainty',
        'proxy_assumption':'p=mask adherence remains unvalidated as a causal response to spending'},indent=2))
    (out/'complete.json').write_text(json.dumps({'complete':True}))


if __name__=='__main__':main()
