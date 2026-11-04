"""Proposal-aligned temporal + state holdouts and survey alternatives."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from .prediction import predict


def main():
    output = Path('outputs/midterm/prediction')
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv('outputs/multi_month/features_targets.csv', dtype={'fips': str})
    panel = panel[panel.month <= '2021-01'].copy().reset_index(drop=True)
    panel['constant'] = 1.
    panel['calendar'] = pd.PeriodIndex(panel.feature_month, freq='M').asi8
    # State averages use past measured surveys, not current target labels.
    panel['state_previous_mask'] = panel.groupby(['feature_month', 'geo_group']).previous_mask.transform('mean')
    sets = {'mean':['constant'], 'population':['log_population'],
        'twitter':['log_population','mean_sentiment','positive_share','negative_share','log_post_count','sentiment_change'],
        'history':['log_population','previous_mask'],
        'history_twitter':['log_population','previous_mask','mean_sentiment','positive_share','negative_share','log_post_count','sentiment_change'],
        'history_calendar':['log_population','previous_mask','calendar'],
        'history_calendar_twitter':['log_population','previous_mask','calendar','mean_sentiment','positive_share','negative_share','log_post_count','sentiment_change']}
    results = []
    test_mask = pd.to_datetime(panel.target_start) >= pd.Timestamp('2020-12-01')
    for model in ['ridge','forest']:
        predictions, fold = predict(panel, sets, folds=5, model=model, test_start='2020-12-01')
        predictions['month'] = panel.month
        predictions['persistence_predicted'] = panel.previous_mask.where(test_mask)
        predictions['state_history_predicted'] = panel.state_previous_mask.where(test_mask)
        predictions.to_csv(output/f'{model}_state_temporal_predictions.csv', index=False)
        fold.to_csv(output/f'{model}_fold_metrics.csv', index=False)
        for scope in ['all','2020-12','2021-01']:
            keep = test_mask if scope == 'all' else test_mask & (panel.month == scope)
            for name in [*sets, 'persistence', 'state_history']:
                y = panel.loc[keep, 'responsiveness']
                estimate = predictions.loc[keep, name+'_predicted']
                results.append({'model': model, 'scope': scope, 'features': name, 'counties': len(y),
                    'mae': mean_absolute_error(y, estimate), 'r2': r2_score(y, estimate)})
    pd.DataFrame(results).to_csv(output/'summary.csv', index=False)
    panel.to_csv(output/'features_targets.csv', index=False)
    (output/'manifest.json').write_text(json.dumps({'design': 'Five state folds AND forward target cutoff December 1. Train October–November in other states; test December–January in held-out states. Fixed models; no test tuning.',
        'features': sets, 'survey_alternatives': ['previous county masking','previous state masking average'],
        'limitations': 'Retrospective latest-revision CTIS; population is the only Census covariate. State history averages use prior observations in the evaluation cohort. Both target months share the original five-day question definition.'}, indent=2))
    print(pd.DataFrame(results).query("scope == 'all'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
