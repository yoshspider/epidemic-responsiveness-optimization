"""PPO around a feasible analytic prior, selected on earlier proxy labels only.

Same proposal spending dynamics. All policies use identical architecture and
training; their greedy prior uses only their own available behavioral input.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from .proposal_ppo import World,Environment,train,execute
from .midterm_real import inputs


class AnchoredEnvironment(Environment):
    def allocation(self,weights):
        features,_=self.features()
        if getattr(self,'_anchor_week',None)!=self.week:
            p=self.w.effectiveness*self.observed_p();n=self.w.n
            local,remote=self.w.beta[self.week]
            hazard=local*self.i/n+remote*(self.matrix()@(self.i/n))
            slope=self.s*hazard*p/n
            low,high=-60.,float(np.log(max(slope[self.eligible].max(),1e-30)))
            for _ in range(40):
                value=(low+high)/2
                action=np.where(p>0,n/np.maximum(p,1e-12)*np.maximum(0,np.log(np.maximum(slope,1e-30))-value),0)
                action[~self.eligible]=0
                if action.sum()>self.budget:low=value
                else:high=value
            action*=self.budget/max(action.sum(),1e-30)
            population=np.zeros(len(n));population[self.eligible]=self.budget*n[self.eligible]/n[self.eligible].sum()
            action=.999*action+.001*population  # nonzero support for all eligible counties
            self._anchor=np.log(np.maximum(action[self.eligible]/self.budget,1e-30))
            self._anchor_week=self.week
        result=np.zeros(len(self.w.n))
        result[self.eligible]=self.budget*softmax(self._anchor+features[self.eligible]@weights)
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=4096);args=parser.parse_args()
    torch.set_num_threads(1)
    folder=Path(f'outputs/anchored_ppo_seed{args.seed}');folder.mkdir(exist_ok=True)
    initial,behavior,_,matrices=inputs();n=initial['n'];eligible=behavior['eligible']
    counties=pd.read_csv('outputs/midterm/real/counties.csv',dtype={'fips':str})
    masks=pd.read_csv('outputs/multi_month/mask_county_month.csv',dtype={'fips':str})
    masks=masks[masks.signal=='smoothed_wwearing_mask']
    truth=masks[masks.month=='2020-10'].set_index('fips').responsiveness.reindex(counties.fips).fillna(0).to_numpy()
    prior=float(truth[eligible].mean())
    survey=masks[masks.month=='2020-09'].set_index('fips').responsiveness.reindex(counties.fips).fillna(prior).to_numpy()
    forecasts=pd.read_csv('outputs/tbcov_prediction/forest_out_of_fold_predictions.csv',dtype={'fips':str}).set_index('fips').reindex(counties.fips)
    twitter=forecasts.sentiment_predicted.fillna(prior).to_numpy()
    (folder/'manifest.json').write_text(json.dumps({'seed':args.seed,'steps':args.steps,
        'method':'PPO residual scores around analytic myopic prior plus 0.1% population allocation for support; identical rules across information levels',
        'training':'October measured proxy + SD.02 simulated uncertainty; initial state randomization and rates .13–.27; first-week mobility only; no November/December response labels',
        'selection':'zero residual and checkpoints every 2048 decisions; select minimum expected infections on eight fixed, disjoint October-based validation worlds; no held-out evaluation selection',
        'regularization':{'latent_mean_squared_penalty':.01,'initial_latent_log_sd':-2.},
        'prior':prior,'network_counties':len(n),'eligible_counties':int(eligible.sum())},indent=2))
    for name,info,prediction in [('blind','blind',twitter),('aggregate','aggregate',twitter),
                                  ('oracle','oracle',twitter),('twitter','predicted',twitter),('survey','predicted',survey)]:
        if (folder/f'{name}.complete.json').exists():continue
        def factory(seed):
            rng=np.random.default_rng(seed)
            infected=np.minimum(np.rint(initial['i']*rng.uniform(.7,1.3)),n-initial['r'])
            p=truth.copy();p[eligible]=np.clip(p[eligible]+rng.normal(0,.02,eligible.sum()),0,1)
            beta=rng.uniform(.13,.27)
            return World(n,n-infected-initial['r'],infected,initial['r'],np.tile(p,(8,1)),
                np.tile(prediction,(8,1)),matrices[0],np.tile([.85*beta,.15*beta],(8,1)),
                float(rng.choice([.02,.10])),prior,eligible,weeks=8)
        best=[np.inf];scores=[]
        def validate(model,update):
            values=[execute(factory(990000+k),info,model,stochastic=False,environment_type=AnchoredEnvironment)[0] for k in range(8)]
            average=float(np.mean(values));scores.append({'update':update,'mean_validation_infections':average})
            if average<best[0]:
                best[0]=average;torch.save(model.state_dict(),folder/f'{name}_selected.pt')
            pd.DataFrame(scores).to_csv(folder/f'{name}_validation.csv',index=False)
        def initialize(model):
            with torch.no_grad():
                model.actor[-1].weight.zero_();model.actor[-1].bias.zero_();model.log_std.fill_(-2.)
            validate(model,0)
        train(info,None,prior,args.seed,args.steps,folder/f'{name}_last.pt',world_factory=factory,
            environment_type=AnchoredEnvironment,initialize=initialize,evaluation_callback=validate,actor_penalty=.01)
        (folder/f'{name}.complete.json').write_text(json.dumps({'complete':True,'selected_update':min(scores,key=lambda row:row['mean_validation_infections'])['update']}))
    (folder/'complete.json').write_text(json.dumps({'complete':True}))


if __name__=='__main__':main()
