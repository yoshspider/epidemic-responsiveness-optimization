"""Audit actual publication vintages and compare timely versus stale surveys."""
import json
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from .acquire import download


def main():
    out=Path('outputs/survey_asof');out.mkdir(exist_ok=True)
    raw=Path('data/raw/survey_asof');raw.mkdir(exist_ok=True)
    counties=pd.read_csv('outputs/midterm/real/counties.csv',dtype={'fips':str})
    latest=pd.concat([pd.read_csv(f'data/raw/ctis_mask_2020-{month:02d}_daily.csv',dtype={'geo_value':str}) for month in [11,12]])
    latest=latest[(latest.missing_value==0)&(~latest.geo_value.str.endswith('000'))].copy()
    table=latest.pivot(index='geo_value',columns='time_value',values='value')/100
    truth=[];lag3=[];lag14=[];audits=[]
    for date in pd.date_range('2020-11-02',periods=8,freq='7D'):
        day=int(date.strftime('%Y%m%d'));d3=int((date-pd.Timedelta(days=3)).strftime('%Y%m%d'));d14=int((date-pd.Timedelta(days=14)).strftime('%Y%m%d'))
        path=raw/f'{day}.csv';metadata=path.with_suffix('.source.json')
        url='https://api.delphi.cmu.edu/epidata/covidcast/?'+urlencode({'data_source':'fb-survey','signal':'smoothed_wwearing_mask',
            'time_type':'day','geo_type':'county','time_values':f'{d3},{d14}','as_of':str(day),'geo_value':'*','format':'csv'})
        if not path.exists():
            record=download(url,path);metadata.write_text(json.dumps(record,indent=2))
        data=pd.read_csv(path,dtype={'geo_value':str})
        if data.empty or data.issue.max()>day:raise ValueError('Missing vintage or publication after decision')
        if not set(data.time_value).issubset({d3,d14}):raise ValueError('Unexpected measurement date')
        data=data[(data.missing_value==0)&(~data.geo_value.str.endswith('000'))]
        def values(when):return data[data.time_value==when].set_index('geo_value').value.reindex(counties.fips).to_numpy()/100
        truth.append(table.reindex(counties.fips)[day].to_numpy())
        lag3.append(values(d3));lag14.append(values(d14))
        audits.append({'decision':day,'lag3_measurement':d3,'lag14_measurement':d14,'max_issue':int(data.issue.max()),'rows':len(data)})
        print(f'As-of decision {day}: {len(data)} published county estimates; no future issue',flush=True)
    truth=np.asarray(truth);lag3=np.asarray(lag3);lag14=np.asarray(lag14)
    eligible=np.isfinite(truth).all(0)&np.isfinite(lag3).all(0)&np.isfinite(lag14).all(0)
    np.savez_compressed(out/'behavior.npz',truth=np.nan_to_num(truth),lag3=np.nan_to_num(lag3),lag14=np.nan_to_num(lag14),eligible=eligible)
    rows=[]
    for name,pred in [('lag3',lag3),('lag14',lag14)]:
        actual=truth[:,eligible].ravel();estimated=pred[:,eligible].ravel()
        rows.append({'predictor':name,'eligible_counties':int(eligible.sum()),'population':float(counties.population.to_numpy()[eligible].sum()),
                     'mae':float(abs(actual-estimated).mean()),'r2':float(1-((actual-estimated)**2).sum()/((actual-actual.mean())**2).sum())})
    pd.DataFrame(rows).to_csv(out/'prediction_metrics.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'design':'Operational nowcast of a Monday rolling survey proxy using county data actually issued by that Monday; weekly truth is the latest-revised Monday proxy',
        'dates':audits,'eligibility':'Identical complete observed cohort for truth and both historical vintages for all eight weeks; no imputed targets',
        'lag3_limitation':'Three-day-lag measurements overlap the current rolling survey window; high nowcast accuracy is not independent future-behavior forecasting',
        'lag14':'Fourteen-day measurement lag gives non-overlapping seven-day survey pools and five-day recall support; publication availability still checked as of decision',
        'remaining_limitation':'Survey proxy and latest revisions are not causal responsiveness or directly observed spending effects'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
