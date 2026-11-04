"""Public data for a March–April 2021 recommendation-responsiveness experiment."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pandas as pd
from .acquire import download
from .midterm_mobility import one as mobility_download


SIGNALS={'acceptance':'smoothed_waccept_covid_vaccine','recommendation':'smoothed_wvaccine_likely_govt_health'}


def survey(item):
    name,signal,month=item
    folder=Path('data/raw/vaccine_experiment');folder.mkdir(exist_ok=True)
    path=folder/f'{name}_{month}.csv';meta=path.with_suffix('.source.json')
    if path.exists() and meta.exists():return
    period=pd.Period(month,freq='M')
    url=f'https://api.delphi.cmu.edu/epidata/covidcast/?data_source=fb-survey&signal={signal}&time_type=day&geo_type=county&time_values={period.start_time:%Y%m%d}-{period.end_time:%Y%m%d}&geo_value=*&format=csv'
    record=download(url,path)
    check=pd.read_csv(path,nrows=2)
    if 'geo_value' not in check:raise ValueError('Invalid survey response')
    meta.write_text(json.dumps(record,indent=2))
    print('Survey downloaded',name,month,flush=True)


def cases():
    folder=Path('data/raw/vaccine_experiment');folder.mkdir(exist_ok=True)
    path=folder/'nyt_2021.csv';meta=path.with_suffix('.source.json')
    if path.exists() and meta.exists():return
    url='https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2021.csv'
    record=download(url,path);meta.write_text(json.dumps(record,indent=2))
    print('2021 case data downloaded',flush=True)


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(cases)]
        futures += [pool.submit(mobility_download,date) for date in pd.date_range('2021-03-01',periods=8,freq='7D')]
        futures += [pool.submit(survey,(name,signal,month)) for name,signal in SIGNALS.items() for month in ['2021-02','2021-03','2021-04']]
        for future in futures:future.result()


if __name__=='__main__':main()
