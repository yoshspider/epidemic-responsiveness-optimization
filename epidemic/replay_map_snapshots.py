"""Record weekly county outputs for the retrospective replay, using matched map budget."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from .observed_study import prepare
from .model import Config,Model
from .experiments import run


def main():
    out=Path('outputs/replay_map')
    out.mkdir(parents=True,exist_ok=True)
    counties,n,_,initial,mobility,_,audit=prepare(Path('data/transformed'))
    rates=pd.read_csv('outputs/temporal_replay_fixed_share/weekly_rates.csv')
    schedule=tuple(rates[['day','beta_local','beta_mobility']].itertuples(index=False,name=None))
    cfg=Config(prior=float(counties.nyt_mask_comparison.mean()),budget=float(n.sum()*.01),beta_schedule=schedule)
    model=Model(n,counties.nyt_mask_comparison.to_numpy(),np.ones(len(n)),mobility,cfg)
    metrics=[]
    for policy in ('none','epidemic','oracle'):
        m,_,trajectory,_=run(model,initial,policy,84,0)
        metrics.append(m)
        frame=pd.DataFrame(trajectory)
        frame['fips']=frame.county_index.map(dict(enumerate(counties.fips)))
        frame['week']=np.floor((frame.day-1)/7).astype(int)+1
        weekly=frame.groupby(['week','fips']).agg(incidence=('new_general','sum'),active=('infected','last')).reset_index()
        weekly.to_csv(out/(policy+'_weekly.csv'),index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)


if __name__=='__main__':
    main()
