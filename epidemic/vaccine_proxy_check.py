"""Compare observed county variation in acceptance and masking on common dates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acquire import download


def main():
    folder=Path('data/raw/vaccine_proxy');folder.mkdir(exist_ok=True)
    out=Path('outputs/vaccine_proxy');out.mkdir(exist_ok=True)
    signals=['smoothed_waccept_covid_vaccine','smoothed_wcovid_vaccinated_or_accept','smoothed_wwearing_mask']
    pieces=[];sources=[]
    for signal in signals:
        path=folder/(signal+'.csv')
        url='https://api.delphi.cmu.edu/epidata/covidcast/?data_source=fb-survey&signal='+signal+'&time_type=day&geo_type=county&time_values=20210106-20210131&geo_value=*&format=csv'
        record=download(url,path)
        sources.append(record)
        data=pd.read_csv(path,dtype={'geo_value':str})
        data=data[(data.missing_value==0)&(~data.geo_value.str.endswith('000'))].copy()
        data['p']=pd.to_numeric(data.value)/100
        if not data.p.between(0,1).all():raise ValueError('Invalid survey percent')
        monthly=data.groupby('geo_value').agg(p=('p','mean'),observed_days=('p','size'))
        monthly=monthly[monthly.observed_days>=20].reset_index().rename(columns={'geo_value':'fips','p':signal})
        pieces.append(monthly[['fips',signal]])
        print(signal,len(monthly),'counties with >=20 days',flush=True)
    common=pieces[0]
    for table in pieces[1:]:common=common.merge(table,on='fips',validate='one_to_one')
    population=pd.read_csv('data/transformed/census_population_2019.csv',dtype={'fips':str})
    common=common.merge(population[['fips','population']],on='fips',validate='one_to_one')
    rows=[]
    for signal in signals:
        values=common[signal].to_numpy();weights=common.population.to_numpy()
        mean=np.average(values,weights=weights);sd=np.sqrt(np.average((values-mean)**2,weights=weights))
        rows.append({'signal':signal,'common_counties':len(values),'population':float(weights.sum()),
            'mean':values.mean(),'sd':values.std(ddof=1),'p10':np.quantile(values,.1),'p90':np.quantile(values,.9),
            'population_weighted_mean':mean,'population_weighted_sd':sd,'coefficient_variation':sd/mean})
    common.to_csv(out/'common_counties.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'variation_comparison.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'source_downloads':sources,'dates':['2021-01-06','2021-01-31'],
        'coverage':'same counties with >=20 valid daily estimates in all three signals; monthly descriptive averages of overlapping smoothed estimates',
        'acceptance_definition':'definitely/probably accept a COVID vaccine if offered, among unvaccinated from Jan6 onward',
        'composite_definition':'already vaccinated or definitely/probably accept, among all respondents',
        'limitations':'Stated acceptance is not measured marginal response to spending, and is not automatically masking responsiveness. These January data must not be used prospectively in November 2020.'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
