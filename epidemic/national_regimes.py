"""Declared counterfactual regimes on empirical county inputs. Save every setting."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .model import Model,Config
from .observed_study import prepare
from .experiments import run


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path('outputs/national_regimes'))
    p.add_argument("--days",type=int,default=84)
    p.add_argument("--seeds",type=int,default=0,help="Zero deterministic screen; positive paired stochastic seeds")
    p.add_argument("--confirm-selected",action="store_true",help="Freeze reference/moderate/strong screening regimes and use independent seeds")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    counties,n,_,initial,mobility,_,audit=prepare(Path('data/transformed'))
    fit=json.loads(Path('outputs/observed_study/calibration.json').read_text())
    response=counties.nyt_mask_comparison.to_numpy()
    design=[{'transmission_multiplier':mult,'budget_fraction':budget,'efficacy':efficacy}
            for mult in (1.,1.5,2.) for budget in (.005,.02,.10) for efficacy in (.5,.8)]
    if a.confirm_selected:
        if not a.seeds:
            p.error("Selected confirmation requires positive seeds")
        design=[{'transmission_multiplier':1.,'budget_fraction':.02,'efficacy':.5},
                {'transmission_multiplier':1.5,'budget_fraction':.02,'efficacy':.5},
                {'transmission_multiplier':2.,'budget_fraction':.1,'efficacy':.8}]
    base=Config(beta_local=fit['beta_local'],beta_mobility=fit['beta_mobility'],
                prior=float(response.mean()),mean_prior=float(response.mean()),stochastic=a.seeds>0)
    (a.output/'manifest.json').write_text(json.dumps({'data_kind':'COUNTERFACTUAL TRANSMISSION/INTERVENTION REGIMES ON EMPIRICAL INPUTS',
        'config':asdict(base),'design':design,'days':a.days,'seeds':a.seeds if a.seeds else 1,
        'calibrated_transmission_multiplier':1,'other_multipliers':'hypothetical conditions; not new fits',
        'budget_units':'weekly population-equivalent effort fraction; assumed uniform costs, not actual dollars',
        'input_audit':audit,'confirm_selected':a.confirm_selected,
        'selection':'Reference, moderate-growth and largest screen-effect regime; independently seeded confirmation; selected exploratory findings, not preregistered causal inference' if a.confirm_selected else 'Full fixed 18-setting grid'},indent=2))
    records=pd.read_csv(a.output/'trials.csv').to_dict('records') if a.resume and (a.output/'trials.csv').exists() else []
    completed={(int(row['scenario']),int(row['seed'])) for row in records}
    for scenario,changes in enumerate(design):
        cfg=Config(**{**asdict(base),'beta_local':base.beta_local*changes['transmission_multiplier'],
            'beta_mobility':base.beta_mobility*changes['transmission_multiplier'],
            'budget':float(n.sum()*changes['budget_fraction']),'efficacy':changes['efficacy']})
        model=Model(n,response,np.ones(len(n)),mobility,cfg)
        for seed in range(a.seeds if a.seeds else 1):
            if (scenario,seed) in completed:
                continue
            row={'scenario':scenario,'seed':seed,**changes}
            actions={}
            for policy in ('none','epidemic','oracle'):
                m,_,_,action=run(model,initial,policy,a.days,seed+(7000 if a.confirm_selected else 6000),record=False)
                row[policy+'_infections']=m['infections']
                row[policy+'_spent']=m['spent']
                actions[policy]=action
            row['information_gain']=row['epidemic_infections']-row['oracle_infections']
            row['relative_information_gain_percent']=100*row['information_gain']/row['epidemic_infections']
            row['intervention_gain']=row['none_infections']-row['oracle_infections']
            union=((actions['epidemic']>0)|(actions['oracle']>0)).sum()
            row['jaccard']=float(((actions['epidemic']>0)&(actions['oracle']>0)).sum()/union) if union else 1
            row['allocation_l1']=float(abs(actions['epidemic']-actions['oracle']).sum())
            records.append(row)
        pd.DataFrame(records).to_csv(a.output/'trials.csv',index=False)
        print(f"Regime {scenario+1}/{len(design)} complete: {changes}",flush=True)
    summary=[]
    for scenario,group in pd.DataFrame(records).groupby('scenario'):
        gain=group.information_gain
        half=t.ppf(.975,len(gain)-1)*gain.std(ddof=1)/np.sqrt(len(gain)) if len(gain)>1 else np.nan
        summary.append({'scenario':scenario,**{key:group.iloc[0][key] for key in ('transmission_multiplier','budget_fraction','efficacy')},
            'mean_information_gain':gain.mean(),'ci_low':gain.mean()-half,'ci_high':gain.mean()+half,
            'relative_information_gain_percent':group.relative_information_gain_percent.mean(),
            'mean_intervention_gain':group.intervention_gain.mean(),'mean_jaccard':group.jaccard.mean()})
    pd.DataFrame(summary).to_csv(a.output/'summary.csv',index=False)


if __name__=='__main__':
    main()
