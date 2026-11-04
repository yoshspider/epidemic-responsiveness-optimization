"""Independent survey comparison and clustered prediction contrasts."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    out=Path('outputs/midterm/prediction')
    nyt=pd.read_csv('data/raw/nyt_mask_use.csv',dtype={'COUNTYFP':str})
    nyt['fips']=nyt.COUNTYFP.str.zfill(5)
    nyt['nyt_proxy']=nyt.ALWAYS+nyt.FREQUENTLY+.5*nyt.SOMETIMES
    ctis=pd.read_csv('outputs/multi_month/mask_county_month.csv',dtype={'fips':str})
    ctis=ctis[(ctis.observed_days>=20)&(ctis.signal=='smoothed_wwearing_mask')]
    pairs=ctis.merge(nyt[['fips','nyt_proxy']],on='fips',validate='many_to_one')
    pairs['state']=pairs.fips.str[:2]
    rows=[]
    rng=np.random.default_rng(970001)
    for month,group in pairs.groupby('month'):
        if month>'2021-01': continue
        groups=list(group.groupby('state'))
        bootstrap=[]
        for _ in range(2000):
            sample=pd.concat([groups[i][1] for i in rng.integers(len(groups),size=len(groups))])
            bootstrap.append(sample.nyt_proxy.corr(sample.responsiveness))
        lo,hi=np.quantile(bootstrap,[.025,.975])
        centered=group[['nyt_proxy','responsiveness']]-group.groupby('state')[['nyt_proxy','responsiveness']].transform('mean')
        rows.append({'ctis_month':month,'counties':len(group),'states':len(groups),
            'pearson':group.nyt_proxy.corr(group.responsiveness),
            'spearman':spearmanr(group.nyt_proxy,group.responsiveness).statistic,
            'state_cluster_ci_low':lo,'state_cluster_ci_high':hi,
            'within_state_pearson':centered.nyt_proxy.corr(centered.responsiveness)})
    pd.DataFrame(rows).to_csv(out/'independent_survey_agreement.csv',index=False)
    results=[]
    for model in ['ridge','forest']:
        data=pd.read_csv(out/f'{model}_state_temporal_predictions.csv',dtype={'fips':str})
        data=data[data.history_predicted.notna()]
        data['state']=data.fips.str[:2]
        for baseline in ['twitter','persistence','mean','state_history']:
            gain=abs(data.responsiveness-data[baseline+'_predicted'])-abs(data.responsiveness-data.history_predicted)
            grouped=pd.DataFrame({'state':data.state,'gain':gain}).groupby('state').gain.agg(['sum','count'])
            select=rng.integers(len(grouped),size=(5000,len(grouped)))
            draws=grouped['sum'].to_numpy()[select].sum(1)/grouped['count'].to_numpy()[select].sum(1)
            lo,hi=np.quantile(draws,[.025,.975])
            results.append({'model':model,'history_vs':baseline,'mae_reduction':gain.mean(),
                            'state_cluster_ci_low':lo,'state_cluster_ci_high':hi})
    pd.DataFrame(results).to_csv(out/'history_cluster_comparisons.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__': main()
