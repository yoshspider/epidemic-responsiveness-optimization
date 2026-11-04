"""Acquire public national inputs without bundling raw data in the repository."""
import argparse
import csv
import json
from pathlib import Path
import pandas as pd
from .acquire import download
from .prepare import census_population, nyt_daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mobility',
        action='store_true',
        help='Also download twelve weekly GeoDS files')
    args = parser.parse_args()
    raw = Path('data/raw')
    transformed = Path('data/transformed')
    raw.mkdir(parents=True, exist_ok=True)
    transformed.mkdir(parents=True, exist_ok=True)
    sources = {
        'nyt_cases_2020.csv': 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties-2020.csv',
        'nyt_mask_use.csv': 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/mask-use/mask-use-by-county.csv',
        'census_co_est2019.csv': 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv',
        'census_counties_2020.zip': 'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip',
    }
    provenance = []
    for name, url in sources.items():
        path = raw / name
        if not path.exists():
            provenance.append(download(url, path))
    if provenance:
        (raw / 'bootstrap_downloads.json').write_text(json.dumps(provenance, indent=2))
    population = census_population(pd.read_csv(raw / 'census_co_est2019.csv', encoding='latin-1'))
    population.to_csv(transformed / 'census_population_2019.csv', index=False)
    reports, audit = nyt_daily(pd.read_csv(raw / 'nyt_cases_2020.csv', dtype={'fips': str}))
    reports.to_csv(transformed / 'nyt_daily_reports.csv', index=False)
    (transformed / 'nyt_bootstrap_audit.json').write_text(json.dumps(audit, indent=2))
    from .cdc_alternative import main as svi_download
    if not (raw / 'alternative_sources/cdc_svi2018_counties.csv').exists():
        svi_download()
    from .national_hesitancy_source import main as hhs_download
    hhs_folder = raw / 'national_hesitancy'
    if not (hhs_folder / 'hhs_county_hesitancy_2021.csv').exists():
        hhs_download()
    fields = ['FIPS Code', 'County Name', 'State', 'Estimated hesitant',
              'Estimated hesitant or unsure', 'Estimated strongly hesitant']
    with (hhs_folder / 'hhs_county_hesitancy_2021.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [{key: row[key] for key in fields} for row in csv.DictReader(stream)]
    for row in rows:
        row['FIPS Code'] = row['FIPS Code'].zfill(5)
    with (hhs_folder / 'county_hesitancy_compact.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.mobility:
        from .us_proxy_data import main as mobility_download
        mobility_download()
    print('National survey, Census and case inputs prepared. Raw data remain local.')


if __name__ == '__main__':
    main()
