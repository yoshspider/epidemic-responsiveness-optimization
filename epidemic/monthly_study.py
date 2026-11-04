"""Durable local workflow: download, aggregate, forward validation, and report.

Run from the workspace with tools/bundled_runner.py on this Windows host.
Progress and failures are saved; no need to keep an interactive shell open.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import pandas as pd
from .monthly_sentiment import process
from .monthly_prediction import build_panel, evaluate


FOLDER = Path('outputs/multi_month')
MONTHS = ['2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03']


def run_module(module, arguments, name):
    command = [sys.executable, 'tools/bundled_runner.py', module, *arguments]
    with (FOLDER/(name+'.log')).open('a', encoding='utf-8') as log:
        log.write('\nRUN '+json.dumps(command)+'\n')
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{module} exited {result.returncode}; inspect {name}.log')


def report():
    metrics = pd.read_csv(FOLDER/'temporal_holdout/metrics.csv')
    intervals = pd.read_csv(FOLDER/'temporal_holdout/state_cluster_intervals.csv')
    manifest = json.loads((FOLDER/'temporal_holdout/manifest.json').read_text())
    lines = ['# Multi-month masking prediction', '',
        'Generated '+datetime.now(timezone.utc).isoformat()+'.', '',
        'Complete county sentiment aggregates cover August 2020–March 2021. CTIS targets cover September 2020–April 2021. Targets and previous-month mask estimates require at least 20 observed days; target values are never imputed.', '',
        'Training target months: '+', '.join(manifest['training_months'])+'. Test target months: '+', '.join(manifest['test_months'])+'.', '',
        f"Training rows: {manifest['training_rows']}; test rows: {manifest['test_rows']}.", '',
        'The CTIS questionnaire changes from five-day to seven-day masking recall in February 2021. Stable-definition results are reported separately. This is temporal validation using revised historical data, not verified real-time publication availability.', '',
        '| Predictor | MAE (percentage points) | RMSE (percentage points) | R² |',
        '|---|---:|---:|---:|']
    for row in metrics[metrics.scope == 'all'].itertuples():
        lines.append(f'| {row.predictor} | {100*row.mae:.3f} | {100*row.rmse:.3f} | {row.r2:.3f} |')
    lines += ['', '## Incremental Twitter value', '',
        'Positive paired MAE reductions mean combined sentiment/activity/history features improve prediction. State-cluster bootstrap intervals are conditional on the fitted predictors, exploratory and unadjusted.', '',
        '| Model | Comparator | MAE reduction (points) | 95% interval (points) |',
        '|---|---|---:|---:|']
    for row in intervals.itertuples():
        lines.append(f'| {row.model} | {row.baseline} | {100*row.mae_reduction:.3f} | {100*row.ci_low:.3f}–{100*row.ci_high:.3f} |')
    lines += ['', 'See `metrics.csv` for each test month; pooled R² can conceal temporal changes. No predictor was tuned on these test months. See `stable_definition_holdout`, `temporal_holdout_no_smoothing_overlap`, and `rolling` for sensitivity results.', '',
              'These models predict reported mask adherence. They do not identify responsiveness to marginal intervention spending or establish an allocation benefit.']
    (FOLDER/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    status = {'stage': 'downloading', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'requested_feature_months': MONTHS, 'target_months': 'September 2020–April 2021',
        'source': 'https://crisisnlp.qcri.org/tbcov',
        'scope': 'All complete monthly records for the eight months relevant to the CTIS masking period; not the full fourteen-month release',
        'failures': [], 'completed_aggregates': []}
    lock = threading.Lock()
    def save():
        with lock:
            status['updated_utc'] = datetime.now(timezone.utc).isoformat()
            progress = []
            for month in MONTHS:
                date = datetime.strptime(month, '%Y-%m')
                archive = Path('data/raw')/('tbcov_'+date.strftime('%B%Y').lower()+'.zip')
                checkpoint = archive.with_suffix('.ranges.json')
                if archive.exists():
                    progress.append({'month': month, 'complete': True, 'bytes': archive.stat().st_size})
                elif checkpoint.exists():
                    record = json.loads(checkpoint.read_text())
                    downloaded = sum(min(record['block_size'], record['bytes']-index*record['block_size']) for index in record['completed'])
                    progress.append({'month': month, 'complete': False,
                                     'verified_downloaded_bytes': downloaded, 'total_bytes': record['bytes']})
                else:
                    progress.append({'month': month, 'complete': False, 'queued': True})
            status['downloads'] = progress
            temp = FOLDER/'status.json.tmp'
            temp.write_text(json.dumps(status, indent=2))
            temp.replace(FOLDER/'status.json')
    def queue(months):
        for month in months:
            error = None
            for attempt in range(3):
                try:
                    run_module('epidemic.monthly_download', [month, '--workers', '16'], 'downloads_'+month)
                    error = None
                    break
                except RuntimeError as exception:
                    error = str(exception)
                    time.sleep(10)
            if error:
                with lock:
                    status['failures'].append({'stage': 'download', 'month': month, 'error': error})
                save()
    save()
    try:
        run_module('epidemic.monthly_masks', [], 'mask_download')
        with ThreadPoolExecutor(max_workers=2) as pool:
            downloads = [pool.submit(queue, ['2020-10', '2020-12', '2021-02', '2020-08']),
                         pool.submit(queue, ['2020-11', '2021-01', '2021-03'])]
            remaining = set(MONTHS)
            while remaining:
                did_work = False
                for month in sorted(remaining.copy()):
                    if process(month, FOLDER):
                        remaining.remove(month)
                        status['completed_aggregates'].append(month)
                        did_work = True
                        save()
                if all(future.done() for future in downloads):
                    for future in downloads:
                        future.result()
                    if remaining and not did_work:
                        raise RuntimeError('Downloads ended without all complete archives: '+', '.join(sorted(remaining)))
                if remaining and not did_work:
                    status['stage'] = 'downloading_and_aggregating'
                    save()
                    time.sleep(20)
        status['stage'] = 'evaluating'
        save()
        pd.concat([pd.read_csv(FOLDER/f'sentiment_{month}.csv', dtype={'fips': str})
                   for month in MONTHS]).to_csv(FOLDER/'sentiment_county_month.csv', index=False)
        for suffix, exclude in [('', False), ('_no_smoothing_overlap', True)]:
            panel = build_panel(FOLDER, exclude)
            panel.to_csv(FOLDER/('features_targets'+suffix+'.csv'), index=False)
            evaluate(panel, '2021-01-01', FOLDER/('temporal_holdout'+suffix))
            stable = panel[panel.signal == 'smoothed_wwearing_mask']
            evaluate(stable, '2021-01-01', FOLDER/('stable_definition_holdout'+suffix))
            if not exclude:
                for month in sorted(panel.month.unique()):
                    if month < '2020-12':
                        continue
                    evaluate(panel[panel.month <= month], month+'-01', FOLDER/'rolling'/month)
        report()
        status['stage'] = 'complete'
        status['report'] = str(FOLDER/'REPORT.md')
        save()
    except Exception as error:
        status['stage'] = 'failed'
        status['failures'].append({'error': repr(error)})
        save()
        raise


if __name__ == '__main__':
    main()
