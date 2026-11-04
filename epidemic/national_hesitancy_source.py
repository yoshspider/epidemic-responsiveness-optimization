"""Acquire and audit official survey-based national hesitancy estimates."""
import csv
import json
from pathlib import Path
from .acquire import download


def main():
    out = Path('data/raw/national_hesitancy')
    out.mkdir(exist_ok=True)
    path = out / 'hhs_county_hesitancy_2021.csv'
    source = download('https://data.cdc.gov/api/views/q9mh-h2tw/rows.csv?accessType=DOWNLOAD', path)
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    if not rows:
        raise ValueError('Empty source')
    columns = list(rows[0])
    print('Columns:', columns, flush=True)
    candidates = [name for name in columns if 'fips' in name.lower()]
    if not candidates:
        raise ValueError('Not a county hesitancy table')
    fips_field = candidates[0]
    fips = {r[fips_field].zfill(5) for r in rows if r[fips_field]}
    population = list(csv.DictReader(Path('data/transformed/census_population_2019.csv').open()))
    expected = {r['fips'].zfill(5) for r in population}
    source.update(dict(rows=len(rows), unique_county_fips=len(fips), census2019_counties=len(expected),
                       missing_census_fips=sorted(expected - fips), extra_fips=sorted(fips - expected),
                       methodological_status='Modeled small-area county predictions, NOT independent direct survey estimates for every county',
                       survey_dates=['2021-05-26', '2021-06-07'],
                       method='HPS state-level relationship; ACS2019 PUMS PUMA prediction; PUMA-to-county population crosswalk',
                       documentation='https://catalog.data.gov/dataset/vaccine-hesitancy-for-covid-19-county-and-local-estimates'))
    (out / 'source.json').write_text(json.dumps(source, indent=2))
    print(json.dumps(source, indent=2), flush=True)
    print('First row:', {key: value for key,
          value in rows[0].items() if 'Boundary' not in key}, flush=True)


if __name__ == '__main__':
    main()
