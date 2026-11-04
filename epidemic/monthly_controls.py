"""Exploratory controls for December; retain all fixed shuffle seeds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from .monthly_prediction import fit_predictions


def main():
    folder = Path('outputs/multi_month/interim_december')
    panel = pd.read_csv(folder/'features_targets.csv', dtype={'fips': str})
    for column in ['target_start', 'target_end']:
        panel[column] = pd.to_datetime(panel[column])
    columns = ['previous_mask', 'log_population', 'log_post_count', 'log_posts_per_capita',
               'mean_sentiment', 'positive_share', 'negative_share', 'sentiment_change', 'sentiment_missing']
    sentiment = ['mean_sentiment', 'positive_share', 'negative_share', 'sentiment_change', 'sentiment_missing']
    actual = panel[panel.month == '2020-12'].responsiveness.to_numpy()
    output, predictions = [], []
    def assess(frame, name, seed=None, selected_columns=None):
        train = frame[frame.month < '2020-12']
        test = frame[frame.month == '2020-12']
        estimate = fit_predictions(train, test, selected_columns or columns, 'forest', delta=True)
        output.append({'control': name, 'seed': seed, 'mae': mean_absolute_error(actual, estimate)})
        predictions.extend({'control': name, 'seed': seed, 'fips': fips,
                            'actual': truth, 'prediction': value}
                           for fips, truth, value in zip(test.fips, actual, estimate))
    assess(panel, 'actual_sentiment')
    calendar = panel.copy()
    calendar['calendar_index'] = pd.PeriodIndex(calendar.feature_month, freq='M').asi8
    base = ['previous_mask', 'log_population', 'log_post_count', 'log_posts_per_capita', 'sentiment_missing']
    assess(calendar, 'activity_plus_calendar', selected_columns=base+['calendar_index'])
    national = panel.copy()
    levels = {}
    for month in sorted(panel.feature_month.unique()):
        source_month = pd.read_csv(f'outputs/multi_month/sentiment_{month}.csv', dtype={'fips': str})
        total = source_month.post_count.sum()
        levels[month] = {'mean_sentiment': source_month.sentiment_sum.sum()/total,
                         'positive_share': source_month.positive.sum()/total,
                         'negative_share': source_month.negative.sum()/total}
    for column in ['mean_sentiment', 'positive_share', 'negative_share']:
        national[column] = national.feature_month.map({month: values[column] for month, values in levels.items()})
    national['sentiment_change'] = np.nan
    for month in sorted(levels):
        previous_month = str(pd.Period(month, freq='M')-1)
        if previous_month in levels:
            national.loc[national.feature_month == month, 'sentiment_change'] = levels[month]['mean_sentiment']-levels[previous_month]['mean_sentiment']
    assess(national, 'national_sentiment_only')
    training = panel[panel.month < '2020-12']
    testing = panel[panel.month == '2020-12']
    for name, subset in [('historical_mean_mask_change', training),
                         ('latest_training_month_mask_change', training[training.month == training.month.max()])]:
        change = float((subset.responsiveness-subset.previous_mask).mean())
        estimate = np.clip(testing.previous_mask.to_numpy()+change, 0, 1)
        output.append({'control': name, 'seed': None, 'mae': mean_absolute_error(actual, estimate)})
    # Retain real current activity and survey history; freeze only sentiment
    # levels to September and set their time change to zero.
    static = panel.copy()
    source = pd.read_csv('outputs/multi_month/sentiment_2020-09.csv', dtype={'fips': str}).set_index('fips')
    for column in ['mean_sentiment', 'positive_share', 'negative_share']:
        static[column] = static.fips.map(source[column])
    static['sentiment_change'] = 0.
    static['sentiment_missing'] = static.mean_sentiment.isna().astype(float)
    assess(static, 'static_september_sentiment')
    for seed in range(93000, 93010):
        shuffled = panel.copy()
        rng = np.random.default_rng(seed)
        for _, group in shuffled.groupby('feature_month'):
            shuffled.loc[group.index, sentiment] = group[sentiment].to_numpy()[rng.permutation(len(group))]
        assess(shuffled, 'within_month_shuffled_sentiment', seed)
    pd.DataFrame(output).to_csv(folder/'sentiment_controls.csv', index=False)
    pd.DataFrame(predictions).to_csv(folder/'sentiment_control_predictions.csv', index=False)
    (folder/'sentiment_controls_method.json').write_text(json.dumps({
        'status': 'EXPLORATORY POST-INTERIM DIAGNOSTICS, NOT PRIMARY CONFIRMATION',
        'training': 'October–November', 'test': 'December 2020',
        'shuffle': 'Jointly shuffle sentiment columns across counties separately within each feature month; preserve survey history and actual activity; retain all ten fixed seeds',
        'static': 'Freeze sentiment levels to observed September county values, preserve current activity and survey history; sentiment change=0',
        'model': 'Same fixed forest250/leaf8/random_state42 as interim comparison'}, indent=2))
    print(pd.DataFrame(output).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
