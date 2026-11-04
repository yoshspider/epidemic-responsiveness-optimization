"""Stronger baseline: reveal aggregate response but hide county variation."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import torch
from .midterm_real import load_world
from .proposal_ppo import execute


def main():
    torch.set_num_threads(1)
    output=Path('outputs/aggregate_baseline'); output.mkdir(exist_ok=True)
    previous=pd.read_csv('outputs/midterm/real_confirmation/trials.csv')
    rows=[]
    for mapping in ['adherence','headroom']:
        for fraction in [.02,.1]:
            world=load_world(mapping,fraction)
            for seed in range(8):
                outcome,_=execute(world,'aggregate','greedy',960000+seed)
                rows.append({'mapping':mapping,'fraction':fraction,'seed':seed,'policy':'greedy_aggregate','infections':outcome})
                pd.DataFrame(rows).to_csv(output/'aggregate_trials.csv',index=False)
            print(f'Aggregate-matched baseline {mapping}, budget{fraction} complete',flush=True)
    table=pd.concat([previous,pd.DataFrame(rows)])
    summaries=[]
    for (mapping,fraction),group in table.groupby(['mapping','fraction']):
        wide=group.pivot(index='seed',columns='policy',values='infections')
        for policy in ['greedy_oracle','greedy_history','greedy_persistence','greedy_twitter','greedy_combined']:
            gain=wide.greedy_aggregate-wide[policy]
            half=t.ppf(.975,7)*gain.std(ddof=1)/np.sqrt(8)
            summaries.append({'mapping':mapping,'fraction':fraction,'policy':policy,
                'aggregate_baseline_infections':wide.greedy_aggregate.mean(),'gain_vs_aggregate':gain.mean(),
                'ci_low':gain.mean()-half,'ci_high':gain.mean()+half,
                'relative_percent':100*gain.mean()/wide.greedy_aggregate.mean()})
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)


if __name__=='__main__':main()
