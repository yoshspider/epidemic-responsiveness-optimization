"""Actual weekly flows for the eight-week proposal replay; no static fallback."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pandas as pd
from .acquire import download


def one(date):
    stamp = date.strftime('%Y_%m_%d')
    folder = Path('data/raw/midterm_mobility'); folder.mkdir(exist_ok=True)
    path = folder/(stamp+'.csv')
    meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        return
    url = f'https://raw.githubusercontent.com/GeoDS/COVID19USFlows-WeeklyFlows/master/weekly_flows/county2county/weekly_county2county_{stamp}.csv'
    record = download(url, path)
    check = pd.read_csv(path, nrows=2)
    if not {'geoid_o','geoid_d','pop_flows'}.issubset(check):
        raise ValueError('Unexpected mobility source schema')
    meta.write_text(json.dumps(record, indent=2))
    print(f'Weekly mobility {stamp}: {record["bytes"]:,} bytes', flush=True)


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, pd.date_range('2020-11-02', periods=8, freq='7D')))


if __name__ == '__main__':
    main()
