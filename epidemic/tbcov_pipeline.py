"""Aggregate official base TSV ZIP members and evaluate geographic held-out prediction.

No hydration. Crosswalk aliases are explicit; ambiguous names are excluded.
"""
import argparse
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .tbcov import aggregate
from .prediction import predict


def crosswalk(population):
    aliases=[]
    for row in population.itertuples():
        state=row.STNAME.strip().casefold()
        name=row.CTYNAME.strip().casefold()
        aliases.append((row.fips,state,name))
        # Administrative suffix removal is deterministic, documented and collision-checked.
        shortened=re.sub(r' (county|parish|borough|census area|municipality|city and borough|city)$','',name)
        aliases.append((row.fips,state,shortened))
    result=pd.DataFrame(aliases,columns=['fips','state','county']).drop_duplicates()
    ambiguous=result.groupby(['state','county']).fips.nunique()
    bad=set(ambiguous[ambiguous>1].index)
    result=result[[tuple(pair) not in bad for pair in result[['state','county']].to_numpy()]]
    return result,{'ambiguous_aliases_excluded':[list(pair) for pair in sorted(bad)]}


def evaluate(features,folder):
    population=pd.read_csv('data/transformed/census_population_2019.csv',dtype={'fips':str})
    target=pd.read_csv('data/transformed/ctis_october2020_monthly_proxy.csv',dtype={'fips':str})
    targets_before=len(target)
    target=target[target.observed_days>=20].copy()
    frame=target.merge(population,on='fips',validate='one_to_one').merge(features,on='fips',validate='one_to_one',how='left')
    frame['log_population']=np.log(frame.population)
    frame['log_post_count']=np.log1p(frame.post_count.fillna(0))
    frame['constant']=1.
    frame['geo_group']=frame.fips.str[:2]
    frame['feature_end']='2020-09-30'
    frame['target_start']='2020-10-01'
    frame['target_end']='2020-10-31'
    sets={'mean_prior':['constant'],'county_baseline':['log_population'],
          'sentiment':['mean_sentiment','positive_share','negative_share','log_post_count'],
          'combined':['log_population','mean_sentiment','positive_share','negative_share','log_post_count']}
    frame.to_csv(folder/'features_targets.csv',index=False)
    quality=[]
    for model in ('ridge','forest'):
        predictions,metrics=predict(frame,sets,folds=5,model=model)
        predictions.to_csv(folder/(model+'_out_of_fold_predictions.csv'),index=False)
        metrics.to_csv(folder/(model+'_fold_metrics.csv'),index=False)
        for name in sets:
            prediction=predictions[name+'_predicted'].to_numpy()
            actual=frame.responsiveness.to_numpy()
            residual=prediction-actual
            denominator=((actual-actual.mean())**2).sum()
            quality.append({'model':model,'features':name,'counties':len(actual),
                'mae':float(abs(residual).mean()),'rmse':float(np.sqrt((residual**2).mean())),
                'r2':float(1-(residual**2).sum()/denominator)})
    pd.DataFrame(quality).to_csv(folder/'prediction_summary.csv',index=False)
    (folder/'target_coverage.json').write_text(json.dumps({'raw_target_counties':targets_before,
        'minimum_observed_days':20,'retained_counties':len(frame),'counties_with_sentiment':int(frame.mean_sentiment.notna().sum()),
        'target_kind':'monthly mean of daily smoothed weighted CTIS public estimates, not direct pooled survey',
        'feature_window':'September 2020; pilot run may cover only selected member dates; inspect aggregate audit',
        'limitations':'One month, geographic state holdouts; no temporal transfer validation. Missing sentiment imputed within training folds; no county target imputation.'},indent=2))
    print(pd.DataFrame(quality).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,default=Path('data/raw/tbcov_september2020.zip'))
    p.add_argument('--output',type=Path,default=Path('outputs/tbcov_prediction'))
    p.add_argument('--max-members',type=int,default=0,help='0 all members; a positive count creates an explicitly partial pilot')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    population=pd.read_csv('data/transformed/census_population_2019.csv',dtype={'fips':str})
    mapping,aliases=crosswalk(population)
    mapping.to_csv(a.output/'county_alias_crosswalk.csv',index=False)
    audit={'data_kind':'TBCOV EMPIRICAL LABELS WITH CTIS PROXY TARGETS','archive':str(a.archive),
           'partial_pilot':bool(a.max_members),**aliases,'members':[]}
    pieces=[]
    with zipfile.ZipFile(a.archive) as archive:
        members=sorted(name for name in archive.namelist() if name.endswith('.tsv'))
        if a.max_members:
            members=members[:a.max_members]
        for member in members:
            print('Aggregating '+member,flush=True)
            with archive.open(member) as source:
                result,member_audit=aggregate(source,mapping,chunksize=100000,progress=True)
            audit['members'].append({'name':member,**member_audit})
            pieces.append(result)
            table=pd.concat(pieces).groupby('fips')[['post_count','sentiment_sum','positive','negative']].sum().reset_index()
            table['mean_sentiment']=table.sentiment_sum/table.post_count
            table['positive_share']=table.positive/table.post_count
            table['negative_share']=table.negative/table.post_count
            table.to_csv(a.output/'county_sentiment.csv',index=False)
            (a.output/'aggregation_audit.json').write_text(json.dumps(audit,indent=2))
    if not pieces:
        raise ValueError('No TSV base members in source ZIP')
    evaluate(table,a.output)


if __name__=='__main__':
    main()
