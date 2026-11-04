"""Test the fitted constant-rate epidemic model beyond its short holdout."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .observed_study import prepare
from .model import Config,Model


def main():
    folder=Path('outputs/forecast_check')
    folder.mkdir(parents=True,exist_ok=True)
    counties,n,initial,_,mobility,_,audit=prepare(Path('data/transformed'))
    fit=json.loads(Path('outputs/observed_study/calibration.json').read_text())
    cfg=Config(beta_local=fit['beta_local'],beta_mobility=fit['beta_mobility'])
    model=Model(n,counties.nyt_mask_comparison.to_numpy(),np.ones(len(n)),mobility,cfg)
    source=pd.read_csv('data/transformed/nyt_daily_reports.csv',dtype={'fips':str},parse_dates=['date'])
    source=source[source.fips.isin(counties.fips)]
    valid=source.reported_incidence.where((source.reported_incidence>=0)&(~source.date_gap))
    source=source.assign(valid_report=valid)
    observed=source.pivot(index='date',columns='fips',values='valid_report').reindex(columns=counties.fips)
    dates=pd.date_range('2020-07-20','2020-11-22')
    state=initial.copy()
    predicted=[]
    for _ in dates:
        state,new=model.step(state)
        predicted.append(new.sum(1)*.2)
    prediction=np.asarray(predicted)
    observed=observed.reindex(dates).to_numpy()
    records=[]
    for label,start,end in [('training',0,28),('short_holdout',28,42),('long_holdout',42,len(dates))]:
        target=observed[start:end]
        valid=np.isfinite(target)
        records.append({'partition':label,'start':str(dates[start].date()),'end':str(dates[end-1].date()),
            'county_day_rmse':float(np.sqrt(np.mean((prediction[start:end][valid]-target[valid])**2))),
            'observed_reports_valid_cells':float(np.nansum(target)),
            'predicted_reports_same_cells':float(prediction[start:end][valid].sum())})
    pd.DataFrame(records).to_csv(folder/'forecast_metrics.csv',index=False)
    pd.DataFrame({'date':dates,'observed_reports':np.nansum(observed,axis=1),
                  'predicted_reports':np.sum(np.where(np.isfinite(observed),prediction,0),axis=1)}).to_csv(folder/'national_series.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':
    main()
