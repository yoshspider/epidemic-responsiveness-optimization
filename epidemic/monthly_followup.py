"""Run calendar diagnostics automatically after the primary workflow finishes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import pandas as pd
from .monthly_calendar import evaluate_calendar


def main():
    folder = Path('outputs/multi_month')
    state_path = folder/'followup_status.json'
    state = {'stage': 'waiting_for_primary', 'analysis': 'secondary calendar controls'}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2))
    save()
    try:
        for _ in range(4320):
            primary = json.loads((folder/'status.json').read_text())
            if primary['stage'] == 'failed':
                raise RuntimeError('Primary study failed; inspect status.json')
            if primary['stage'] == 'complete':
                break
            time.sleep(30)
        else:
            raise TimeoutError('Primary study did not finish within 36 hours')
        state['stage'] = 'evaluating'
        save()
        for name in ['temporal_holdout', 'stable_definition_holdout',
                     'temporal_holdout_no_smoothing_overlap', 'stable_definition_holdout_no_smoothing_overlap']:
            source = folder/name
            # Sensitivity folders contain the same target design but their own
            # saved panel is kept at the root of the primary workflow.
            suffix = '_no_smoothing_overlap' if name.endswith('_no_smoothing_overlap') else ''
            panel = pd.read_csv(folder/('features_targets'+suffix+'.csv'), dtype={'fips': str})
            if name.startswith('stable_definition'):
                panel = panel[panel.signal == 'smoothed_wwearing_mask']
            panel.to_csv(source/'features_targets.csv', index=False)
            evaluate_calendar(source)
        metrics = pd.read_csv(folder/'temporal_holdout/calendar_controls/metrics.csv')
        intervals = pd.read_csv(folder/'temporal_holdout/calendar_controls/state_cluster_intervals.csv')
        lines = ['# Secondary calendar controls', '',
            'These diagnostics were added after the exploratory December result. They test whether Twitter adds value beyond survey history, posting activity, and calendar time. They do not replace or retune the original primary models.', '',
            '| Predictor (all include calendar time) | MAE (percentage points) | R² |',
            '|---|---:|---:|']
        for row in metrics[metrics.scope == 'all'].itertuples():
            lines.append(f'| {row.predictor} | {100*row.mae:.3f} | {row.r2:.3f} |')
        lines += ['', '| Model | Sentiment MAE reduction vs history/activity/calendar (points) | State-cluster 95% interval |',
                  '|---|---:|---:|']
        for row in intervals[intervals.baseline.str.contains('lag_activity')].itertuples():
            lines.append(f'| {row.model} | {100*row.mae_reduction:.3f} | {100*row.ci_low:.3f}–{100*row.ci_high:.3f} |')
        (folder/'CALENDAR_CONTROLS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        report = folder/'REPORT.md'
        with report.open('a', encoding='utf-8') as stream:
            stream.write('\n## Additional temporal controls\n\nThe initial model comparisons omit explicit calendar time. See [CALENDAR_CONTROLS.md](CALENDAR_CONTROLS.md) for secondary tests of sentiment beyond history, activity and calendar. In the exploratory December test, shuffling county sentiment preserved the initial gain, and a calendar baseline outperformed that initial sentiment model. Interpret incremental Twitter claims using these controls as well.\n')
        state['stage'] = 'complete'
        save()
    except Exception as error:
        state.update(stage='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
