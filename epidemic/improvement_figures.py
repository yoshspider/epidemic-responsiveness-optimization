"""Standalone scientific figures; no slide deck or report generation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('outputs/improvement_figures');out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    synthetic=pd.read_csv('outputs/proposal_ppo_confirmation/summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for axis,scenario in zip(axes,['moderate','strong']):
        for policy,color,offset in [('ppo_oracle','#236b8e',-.003),('ppo_predicted','#c16b26',.003)]:
            data=synthetic[(synthetic.scenario==scenario)&(synthetic.policy==policy)]
            scale=data.relative_percent/data.gain_vs_blind
            axis.errorbar(100*(data.fraction+offset),data.relative_percent,
                yerr=[(data.gain_vs_blind-data.ci_low)*scale,(data.ci_high-data.gain_vs_blind)*scale],
                marker='o',capsize=4,color=color,label='Known p' if policy=='ppo_oracle' else 'Learned p')
        axis.set(xlabel='Weekly budget / population (%)',title=scenario.capitalize()+' response variation')
        axis.set_xticks([2,10,20]);axis.axhline(0,color='gray',lw=.7);axis.set_ylim(-.5,11)
    axes[0].set_ylabel('Fewer infections versus uninformed PPO (%)')
    axes[1].legend(frameon=False)
    fig.suptitle('Synthetic PPO: 24 fresh worlds, three training seeds, 84 days\nPaired 95% intervals across worlds; seed-averaged effects')
    fig.savefig(out/'synthetic_ppo_confirmation.png',dpi=200);plt.close(fig)
    baseline=pd.read_csv('outputs/aggregate_baseline/summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    names=['greedy_oracle','greedy_history','greedy_twitter']
    for axis,mapping in zip(axes,['adherence','headroom']):
        data=baseline[(baseline.mapping==mapping)&(baseline.fraction==.1)].set_index('policy').reindex(names)
        mean=data.gain_vs_aggregate/1000
        axis.errorbar(np.arange(3),mean,yerr=[mean-data.ci_low/1000,data.ci_high/1000-mean],fmt='o',capsize=5,color='#236b8e')
        axis.set_xticks(range(3),['Known p','Survey history','Twitter'])
        axis.axhline(0,color='gray',lw=.8)
        axis.set_title('p = masking (proposal)' if mapping=='adherence' else 'p = 1 − masking (alternative assumption)')
        axis.set_ylabel('Additional infections prevented (thousands)')
    fig.suptitle('Historical replay: compare against the same aggregate response information\nEight paired epidemic seeds; conditional model uncertainty only; different panel scales')
    fig.savefig(out/'aggregate_matched_allocation.png',dpi=200);plt.close(fig)
    counties=pd.read_csv('outputs/midterm/real/counties.csv',dtype={'fips':str})
    behavior=np.load('outputs/midterm/real/behavior.npz');keep=behavior['eligible']
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    rows=[]
    for month,week,color in [('November',0,'#236b8e'),('December',6,'#c16b26')]:
        p=behavior['truth'][week,keep]
        weights=counties.population.to_numpy()[keep]
        for axis,values in [(axes[0],p),(axes[1],1-p)]:
            axis.hist(values,bins=18,alpha=.5,color=color,label=month)
        for mapping,values in [('adherence',p),('headroom',1-p)]:
            mean=np.average(values,weights=weights)
            sd=np.sqrt(np.average((values-mean)**2,weights=weights))
            rows.append({'month':month,'mapping':mapping,'counties':len(p),
                         'population_weighted_mean':mean,'population_weighted_sd':sd,'coefficient_variation':sd/mean})
    axes[0].set(xlabel='Observed masking fraction',ylabel='Counties',title='High masking: limited relative variation')
    axes[1].set(xlabel='Unmasked fraction',ylabel='Counties',title='Same observations, different response interpretation')
    axes[1].legend(frameon=False)
    fig.savefig(out/'proxy_distribution.png',dpi=200);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'proxy_distribution_stats.csv',index=False)


if __name__=='__main__':main()
