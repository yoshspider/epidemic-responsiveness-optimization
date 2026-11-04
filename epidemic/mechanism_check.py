"""Direct intervention-strength diagnostic; fixed spending, no policy training."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


def allocation(s,i,n,matrix,rates,p,eligible,budget):
    local,remote=rates
    h=local*i/n+remote*(matrix@(i/n));slope=s*h*p/n
    lo,hi=-60.,float(np.log(max(slope[eligible].max(),1e-30)))
    for _ in range(60):
        value=(lo+hi)/2
        a=np.where(p>0,n/np.maximum(p,1e-12)*np.maximum(0,np.log(np.maximum(slope,1e-30))-value),0)
        a[~eligible]=0
        if a.sum()>budget:lo=value
        else:hi=value
    return a*budget/max(a.sum(),1e-30)


def main():
    folder=Path('outputs/midterm/real');out=Path('outputs/mechanism_check');out.mkdir(exist_ok=True)
    data=np.load(folder/'initial.npz');behavior=np.load(folder/'behavior.npz')
    n=data['n'];eligible=behavior['eligible'];true=behavior['truth']
    rates=pd.read_csv(folder/'weekly_rates.csv')[['beta_local','beta_mobility']].to_numpy()
    matrices=[sparse.load_npz(folder/f'mobility_{week}.npz') for week in range(8)]
    records=[]
    for fraction in [.02,.10]:
        a=np.zeros(len(n));a[eligible]=fraction*n[eligible]
        for scenario,value in [('p_zero',0.),('p_half',.5),('p_full',1.),('observed_proxy',None)]:
            s,i,r=data['s'].copy(),data['i'].copy(),data['r'].copy();total=0.
            for week in range(8):
                p=true[week] if value is None else np.full(len(n),value)
                multiplier=np.exp(-a*p/n)
                for day in range(7):
                    h=(rates[week,0]*i/n+rates[week,1]*(matrices[week]@(i/n)))*multiplier
                    new=s*(-np.expm1(-h));recovery=i*(-np.expm1(-1/7))
                    s-=new;i+=new-recovery;r+=recovery;total+=new.sum()
            records.append({'fraction':fraction,'scenario':scenario,'new_infections':total,'active_end':i.sum(),'susceptible_end':s.sum()})
    table=pd.DataFrame(records)
    table['prevented_vs_p_zero']=table.groupby('fraction').new_infections.transform('first')-table.new_infections
    table.to_csv(out/'fixed_spending.csv',index=False)
    diagnostic=[]
    for fraction in [.02,.10]:
        p=true[0];mean=np.average(p[eligible],weights=n[eligible]);budget=fraction*n[eligible].sum()
        blind=allocation(data['s'],data['i'],n,matrices[0],rates[0],np.full(len(n),mean),eligible,budget)
        informed=allocation(data['s'],data['i'],n,matrices[0],rates[0],p,eligible,budget)
        a=blind>1;b=informed>1
        diagnostic.append({'fraction':fraction,'initial_week_budget_reallocated_percent':50*abs(blind-informed).sum()/budget,
            'funded_counties_aggregate':int(a.sum()),'funded_counties_oracle':int(b.sum()),
            'funded_set_jaccard':float((a&b).sum()/(a|b).sum()),'allocation_spearman':spearmanr(blind[eligible],informed[eligible]).statistic})
    pd.DataFrame(diagnostic).to_csv(out/'allocation_change.csv',index=False)
    risk=data['s']/n*(rates[0,0]*data['i']/n+rates[0,1]*(matrices[0]@(data['i']/n)))
    correlations={
        'eligible_counties':int(eligible.sum()),
        'spearman_population_p_november':spearmanr(n[eligible],true[0,eligible]).statistic,
        'spearman_population_p_december':spearmanr(n[eligible],true[6,eligible]).statistic,
        'spearman_epidemiological_marginal_score_p':spearmanr(risk[eligible],true[0,eligible]).statistic,
        'spearman_epidemiological_score_behavior_adjusted_score':spearmanr(risk[eligible],risk[eligible]*true[0,eligible]).statistic,
        'epidemiological_score_p10_p50_p90':np.quantile(risk[eligible],[.1,.5,.9]).tolist(),
        'responsiveness_p10_p50_p90':np.quantile(true[0,eligible],[.1,.5,.9]).tolist()}
    import json
    (out/'priority_correlations.json').write_text(json.dumps(correlations,indent=2))
    print(correlations,flush=True)
    print(table.to_string(index=False),flush=True)
    print(pd.DataFrame(diagnostic).to_string(index=False),flush=True)


if __name__=='__main__':main()
