"""Strict state+time holdouts for CDC/ACS covariates and lagged state surveys."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score


def build_panel():
    panel=pd.read_csv('outputs/multi_month/features_targets.csv',dtype={'fips':str})
    panel=panel[panel.month<='2021-01'].copy().reset_index(drop=True)
    svi=pd.read_csv('data/raw/alternative_sources/cdc_svi2018_counties.csv',dtype={'FIPS':str})
    svi['fips']=svi.FIPS.str.zfill(5)
    columns=['E_PCI','EP_POV','EP_UNEMP','EP_NOHSDP','EP_AGE65','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_NOVEH','EP_UNINSUR','AREA_SQMI']
    for column in columns:svi[column]=pd.to_numeric(svi[column],errors='coerce').where(lambda x:x>=0)
    panel=panel.merge(svi[['fips','ST_ABBR']+columns],on='fips',how='left',validate='many_to_one')
    states=dict(zip(svi.fips.str[:2],svi.ST_ABBR.str.lower()))
    panel['state_abbr']=panel.fips.str[:2].map(states)
    state=pd.read_csv('data/raw/alternative_sources/state_mask_monthly.csv').rename(columns={'geo_value':'state_abbr','month':'feature_month'})
    state=state[state.observed_days>=20]
    panel=panel.merge(state[['state_abbr','feature_month','state_mask']],on=['state_abbr','feature_month'],how='left',validate='many_to_one')
    if panel.state_mask.isna().any():raise ValueError('Missing independent state survey')
    panel['log_density']=np.log1p(panel.population/panel.AREA_SQMI.replace(0,np.nan))
    panel['calendar']=pd.PeriodIndex(panel.feature_month,freq='M').asi8
    return panel,columns[:-1]+['log_density']


def evaluate(panel,demographics,cutoff,output):
    output.mkdir(parents=True,exist_ok=True)
    before=pd.to_datetime(panel.target_end)<pd.Timestamp(cutoff)
    after=pd.to_datetime(panel.target_start)>=pd.Timestamp(cutoff)
    sets={'population':['log_population'],'demographics':['log_population']+demographics,
        'demographics_twitter':['log_population']+demographics+['mean_sentiment','positive_share','negative_share','log_post_count','sentiment_change'],
        'state_demographics':['log_population','state_mask']+demographics,
        'state_demographics_twitter':['log_population','state_mask']+demographics+['mean_sentiment','positive_share','negative_share','log_post_count','sentiment_change'],
        'county_history':['log_population','previous_mask'],
        'county_state_demographics':['log_population','previous_mask','state_mask']+demographics}
    summary=[]; predictions=panel[['fips','month','geo_group','responsiveness','previous_mask','state_mask']].copy()
    for model in ['ridge','forest']:
        for name,columns in sets.items():
            values=np.full(len(panel),np.nan)
            for train,test in GroupKFold(5).split(panel,groups=panel.geo_group):
                train=train[before.iloc[train].to_numpy()]; test=test[after.iloc[test].to_numpy()]
                if not len(train) or not len(test):raise ValueError('Empty state/temporal partition')
                estimator=Ridge(alpha=10) if model=='ridge' else RandomForestRegressor(n_estimators=250,min_samples_leaf=8,random_state=42,n_jobs=2)
                pipeline=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),estimator)
                pipeline.fit(panel.iloc[train][columns],panel.iloc[train].responsiveness)
                values[test]=np.clip(pipeline.predict(panel.iloc[test][columns]),0,1)
            predictions[model+'_'+name]=values
            for scope,keep in [('all',after),('2020-11',after&(panel.month=='2020-11')),('2020-12',after&(panel.month=='2020-12')),('2021-01',after&(panel.month=='2021-01'))]:
                if not keep.any():continue
                summary.append({'model':model,'features':name,'scope':scope,'rows':int(keep.sum()),
                    'mae':mean_absolute_error(panel.loc[keep,'responsiveness'],values[keep]),
                    'r2':r2_score(panel.loc[keep,'responsiveness'],values[keep])})
        print(f'Alternative predictor {model}, cutoff {cutoff} complete',flush=True)
    predictions['persistence']=panel.previous_mask.where(after)
    predictions['state_survey']=panel.state_mask.where(after)
    predictions.to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    panel.to_csv(output/'features_targets.csv',index=False)
    (output/'manifest.json').write_text(json.dumps({'cutoff':cutoff,'training_target_months':sorted(panel.loc[before,'month'].unique()),
        'test_target_months':sorted(panel.loc[after,'month'].unique()),'design':'Five state holdouts AND forward time cutoff; identical rows in all models; all imputation/scaling trained within split; no test tuning',
        'new_information':'Official CDC SVI2018 county covariates derived from 2014–2018 ACS; independent prior-month state CTIS estimates; no current target state survey',
        'feature_sets':sets,'limitations':'Observational adherence targets, not causal spending responsiveness; public revised historical sources, not verified exact as-of revisions'},indent=2))
    print(pd.DataFrame(summary).query("scope == 'all'").to_string(index=False),flush=True)


def main():
    panel,columns=build_panel()
    evaluate(panel,columns,'2020-12-01',Path('outputs/alternative_prediction/future_states'))
    # Historical allocator test: only October target labels train, November/
    # December targets remain unseen. January is excluded from this experiment.
    evaluate(panel[panel.month<='2020-12'].reset_index(drop=True),columns,'2020-11-01',Path('outputs/alternative_prediction/allocation_holdout'))


if __name__=='__main__':main()
