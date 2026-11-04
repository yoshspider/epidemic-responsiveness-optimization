"""Public state masking history, independent of county target availability."""
from pathlib import Path
import json
import pandas as pd
from .acquire import download


def main():
    folder=Path('data/raw/alternative_sources'); folder.mkdir(exist_ok=True)
    url='https://api.delphi.cmu.edu/epidata/covidcast/?data_source=fb-survey&signal=smoothed_wwearing_mask&time_type=day&geo_type=state&time_values=20200901-20201231&geo_value=*&format=csv'
    path=folder/'ctis_state_mask_sep_dec2020.csv'
    record=download(url,path)
    frame=pd.read_csv(path,dtype={'geo_value':str})
    if not {'geo_value','time_value','value'}.issubset(frame):raise ValueError('Unexpected survey response')
    frame=frame[pd.to_numeric(frame.missing_value)==0].copy()
    frame['date']=pd.to_datetime(frame.time_value.astype(str),format='%Y%m%d')
    frame['month']=frame.date.dt.strftime('%Y-%m')
    result=frame.groupby(['geo_value','month']).agg(state_mask=('value','mean'),observed_days=('value','size')).reset_index()
    result['state_mask']/=100
    result.to_csv(folder/'state_mask_monthly.csv',index=False)
    (folder/'ctis_state_source.json').write_text(json.dumps(record,indent=2))
    print(result.groupby('month').size().to_string(),flush=True)


if __name__=='__main__':main()
