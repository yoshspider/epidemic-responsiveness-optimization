"""One targeted alternative: self-reported response to health recommendations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .acquire import download


def main():
    out=Path('outputs/proxy_candidate_check');out.mkdir(exist_ok=True)
    raw=Path('data/raw/proxy_candidate_check');raw.mkdir(exist_ok=True)
    signals=['smoothed_waccept_covid_vaccine','smoothed_wvaccine_likely_govt_health','smoothed_wwearing_mask_7d']
    tables=[];records=[]
    for signal in signals:
        url='https://api.delphi.cmu.edu/epidata/covidcast/?data_source=fb-survey&signal='+signal+'&time_type=day&geo_type=county&time_values=20210301-20210331&geo_value=*&format=csv'
        path=raw/(signal+'.csv');record=download(url,path);records.append(record)
        data=pd.read_csv(path,dtype={'geo_value':str})
        data=data[(data.missing_value==0)&(~data.geo_value.str.endswith('000'))]
        grouped=data.groupby('geo_value').agg(value=('value','mean'),days=('value','size'))
        grouped=grouped[grouped.days>=20]
        tables.append((grouped.value/100).rename(signal))
        print(signal,len(grouped),'counties',flush=True)
    common=pd.concat(tables,axis=1,join='inner').rename_axis('fips').reset_index()
    population=pd.read_csv('data/transformed/census_population_2019.csv',dtype={'fips':str})
    common=common.merge(population[['fips','population']],on='fips',validate='one_to_one')
    rows=[]
    for signal in signals:
        values=common[signal].to_numpy();weights=common.population.to_numpy();mean=np.average(values,weights=weights)
        sd=np.sqrt(np.average((values-mean)**2,weights=weights))
        rows.append({'signal':signal,'counties':len(common),'population':int(weights.sum()),'mean':values.mean(),
            'sd':values.std(ddof=1),'p10':np.quantile(values,.1),'p90':np.quantile(values,.9),
            'weighted_mean':mean,'weighted_sd':sd,'relative_sd':sd/mean})
    common.to_csv(out/'county_values.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'variation.csv',index=False)
    (out/'sources.json').write_text(json.dumps(records,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
