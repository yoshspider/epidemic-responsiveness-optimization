"""Checkpointed complete-archive county/month aggregation; no tweet hydration."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import zipfile
import pandas as pd
from .tbcov import aggregate
from .tbcov_pipeline import crosswalk


def process(month, output):
    date = datetime.strptime(month, '%Y-%m')
    archive_path = Path('data/raw')/('tbcov_'+date.strftime('%B%Y').lower()+'.zip')
    destination = output/f'sentiment_{month}.csv'
    complete = output/f'sentiment_{month}.complete.json'
    if complete.exists() and destination.exists():
        return True
    if month == '2020-09':
        source = Path('outputs/tbcov_prediction')
        audit = json.loads((source/'aggregation_audit.json').read_text())
        if audit['partial_pilot']:
            raise ValueError('Cannot reuse partial September aggregate')
        result = pd.read_csv(source/'county_sentiment.csv', dtype={'fips': str})
        result['month'] = month
        result.to_csv(destination, index=False)
        complete.write_text(json.dumps({'month': month, 'reused_complete_source': str(source),
                                        'audit': audit}, indent=2))
        return True
    if not archive_path.exists():
        return False
    population = pd.read_csv('data/transformed/census_population_2019.csv', dtype={'fips': str})
    mapping, alias_audit = crosswalk(population)
    checkpoint = output/f'members_{month}'
    checkpoint.mkdir(exist_ok=True)
    pieces, members_audit = [], []
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith('.tsv'))
        if not members:
            raise ValueError('No base TSV members')
        for number, member in enumerate(members):
            path = checkpoint/f'{number:03d}.csv'
            audit_path = checkpoint/f'{number:03d}.json'
            info = archive.getinfo(member)
            identity = {'name': member, 'crc': info.CRC, 'uncompressed_bytes': info.file_size}
            if path.exists() and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if audit['identity'] != identity:
                    raise ValueError('Member checkpoint mismatch')
                result = pd.read_csv(path, dtype={'fips': str})
            else:
                print(f'{month}: aggregate member {number+1}/{len(members)} {member}', flush=True)
                with archive.open(member) as source:
                    result, details = aggregate(source, mapping, progress=True)
                if set(result.month) != {month}:
                    raise ValueError('Archive contains unexpected month')
                audit = {'identity': identity, **details}
                result.to_csv(path, index=False)
                audit_path.write_text(json.dumps(audit, indent=2))
            pieces.append(result)
            members_audit.append(audit)
    totals = pd.concat(pieces).groupby(['fips', 'month'])[
        ['post_count', 'sentiment_sum', 'positive', 'negative']].sum().reset_index()
    totals['mean_sentiment'] = totals.sentiment_sum/totals.post_count
    totals['positive_share'] = totals.positive/totals.post_count
    totals['negative_share'] = totals.negative/totals.post_count
    totals.to_csv(destination, index=False)
    complete.write_text(json.dumps({'month': month, 'archive': str(archive_path),
        'complete_all_members': True, 'members': members_audit, **alias_audit}, indent=2))
    print(f'{month}: complete county aggregate, {len(totals)} counties', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--months', nargs='+', default=[f'2020-{i:02d}' for i in range(8,13)]+[f'2021-{i:02d}' for i in range(1,4)])
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/multi_month'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remaining = set(args.months)
    while remaining:
        for month in sorted(remaining.copy()):
            if process(month, args.output):
                remaining.remove(month)
        if remaining:
            print('Waiting for complete archives: '+', '.join(sorted(remaining)), flush=True)
            if not args.wait:
                break
            time.sleep(20)
    pieces = [pd.read_csv(args.output/f'sentiment_{month}.csv', dtype={'fips': str})
              for month in args.months if (args.output/f'sentiment_{month}.complete.json').exists()]
    if pieces:
        pd.concat(pieces).to_csv(args.output/'sentiment_county_month.csv', index=False)


if __name__ == '__main__':
    main()
