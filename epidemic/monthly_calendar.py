"""Secondary calendar controls: does sentiment beat an explicit time baseline?"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .monthly_prediction import fit_predictions, summarize, cluster_intervals


def evaluate_calendar(source):
    panel = pd.read_csv(source/'features_targets.csv', dtype={'fips': str})
    for column in ['target_start', 'target_end']:
        panel[column] = pd.to_datetime(panel[column])
    manifest = json.loads((source/'manifest.json').read_text())
    cutoff = pd.Timestamp(manifest['cutoff'])
    panel['calendar_index'] = pd.PeriodIndex(panel.feature_month, freq='M').asi8
    train = panel[panel.target_end < cutoff]
    test = panel[panel.target_start >= cutoff]
    output = test[['fips', 'geo_group', 'month', 'responsiveness']].copy()
    output['persistence_prediction'] = test.previous_mask
    history = ['previous_mask', 'log_population', 'calendar_index']
    activity = history+['log_post_count', 'log_posts_per_capita', 'sentiment_missing']
    combined = activity+['mean_sentiment', 'positive_share', 'negative_share', 'sentiment_change']
    for model in ['ridge', 'forest', 'boosting']:
        # Shared names permit the same paired state-cluster comparison function.
        for name, columns in [('lag_population_delta', history), ('lag_activity_delta', activity),
                              ('combined_delta', combined)]:
            output[f'{model}_{name}_prediction'] = fit_predictions(train, test, columns, model, delta=True)
    destination = source/'calendar_controls'
    destination.mkdir(exist_ok=True)
    output.to_csv(destination/'predictions.csv', index=False)
    metrics = summarize(output)
    intervals = cluster_intervals(output)
    metrics.to_csv(destination/'metrics.csv', index=False)
    intervals.to_csv(destination/'state_cluster_intervals.csv', index=False)
    (destination/'manifest.json').write_text(json.dumps({
        'analysis_status': 'SECONDARY DIAGNOSTIC ADDED AFTER SEEING DECEMBER INTERIM; NOT ORIGINAL PRIMARY DESIGN',
        'cutoff': str(cutoff.date()), 'training_months': sorted(train.month.unique()),
        'test_months': sorted(test.month.unique()), 'history': history, 'activity': activity,
        'combined': combined, 'rationale': 'Within-month shuffled sentiment also improved the initial December model. Explicit calendar features test whether the apparent sentiment value is a time proxy.',
        'models': 'Identical fixed model settings as original study; all three retained; no hyperparameter tuning',
        'limitations': 'State-cluster intervals conditional on observed months and fitted models, exploratory and unadjusted; no causal attribution'}, indent=2))
    print(metrics[metrics.scope == 'all'].to_string(index=False), flush=True)
    print(intervals.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    evaluate_calendar(args.source)


if __name__ == '__main__':
    main()
