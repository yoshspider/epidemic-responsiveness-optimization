"""Download the CTIS mask series in monthly, auditable requests."""
import json
from pathlib import Path
import time
import pandas as pd
from .acquire import download
from .prepare import ctis_monthly


def main():
    output = Path('outputs/multi_month')
    output.mkdir(parents=True, exist_ok=True)
    pieces, records = [], []
    requests = [(period, 'smoothed_wwearing_mask') for period in pd.period_range('2020-09', '2021-02', freq='M')]
    requests += [(period, 'smoothed_wwearing_mask_7d') for period in pd.period_range('2021-02', '2021-04', freq='M')]
    for period, signal in requests:
        dates = f'{period.start_time:%Y%m%d}-{period.end_time:%Y%m%d}'
        url = 'https://api.delphi.cmu.edu/epidata/covidcast/?data_source=fb-survey&signal='+signal+'&time_type=day&geo_type=county&time_values='+dates+'&geo_value=*&format=csv'
        suffix = '_7d' if signal.endswith('_7d') else ''
        destination = Path('data/raw')/f'ctis_mask_{period}{suffix}_daily.csv'
        metadata = destination.with_suffix('.download.json')
        if not destination.exists():
            for attempt in range(4):
                try:
                    record = download(url, destination)
                    metadata.write_text(json.dumps(record, indent=2))
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(3)
        frame = pd.read_csv(destination, dtype={'geo_value': str})
        monthly, audit = ctis_monthly(frame)
        monthly['signal'] = signal
        pieces.append(monthly)
        # Keep daily observations for a sensitivity excluding cross-month smoothing.
        frame.to_csv(output/f'mask_daily_{period}{suffix}.csv', index=False)
        audit.update({'month': str(period), 'signal': signal, 'source': str(destination), 'url': url,
                      'source_metadata': json.loads(metadata.read_text()) if metadata.exists() else None})
        records.append(audit)
        print(f'CTIS {period}: {len(monthly)} counties', flush=True)
        pd.concat(pieces).to_csv(output/'mask_county_month.csv', index=False)
        (output/'mask_sources.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
