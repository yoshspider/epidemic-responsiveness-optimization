"""Forward-month, paired-panel prediction with persistence and activity ablations.

Historical revised survey values are used; this is temporal validation, not a
validated as-of publication-date forecasting system. No tuning on test months.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_panel(folder, exclude_first_week=False):
    masks = pd.read_csv(folder/'mask_county_month.csv', dtype={'fips': str})
    # February old-definition estimates cover too few days; prefer the new,
    # explicitly labeled series from February onwards.
    masks = masks[~((masks.month >= '2021-02') & (masks.signal == 'smoothed_wwearing_mask'))].copy()
    if exclude_first_week:
        pieces = []
        from .prepare import ctis_monthly
        for (month, signal), _ in masks.groupby(['month', 'signal']):
            suffix = '_7d' if signal.endswith('_7d') else ''
            daily = pd.read_csv(folder/f'mask_daily_{month}{suffix}.csv', dtype={'geo_value': str})
            dates = pd.to_datetime(daily.time_value.astype(str), format='%Y%m%d')
            target, _ = ctis_monthly(daily[dates.dt.day >= 8])
            target['signal'] = signal
            pieces.append(target)
        targets = pd.concat(pieces)
    else:
        targets = masks.copy()
    targets = targets[targets.observed_days >= 20].copy()
    targets['feature_month'] = (pd.PeriodIndex(targets.month, freq='M')-1).astype(str)
    previous = masks[masks.observed_days >= 20][['fips', 'month', 'responsiveness', 'signal']].rename(
        columns={'month': 'feature_month', 'responsiveness': 'previous_mask', 'signal': 'previous_signal'})
    panel = targets.merge(previous, on=['fips', 'feature_month'], how='inner', validate='one_to_one')
    # Every policy uses the same observed-lag cohort; never impute target or lag.
    population = pd.read_csv('data/transformed/census_population_2019.csv', dtype={'fips': str})
    panel = panel.merge(population[['fips', 'population']], on='fips', validate='many_to_one')
    panel['log_population'] = np.log(panel.population)
    panel['geo_group'] = panel.fips.str[:2]
    sentiment = []
    for path in sorted(folder.glob('sentiment_????-??.csv')):
        if path.with_suffix('.complete.json').exists():
            sentiment.append(pd.read_csv(path, dtype={'fips': str}))
    if not sentiment:
        raise ValueError('No completed monthly sentiment aggregates')
    features = pd.concat(sentiment).rename(columns={'month': 'feature_month'})
    if features.duplicated(['fips', 'feature_month']).any():
        raise ValueError('Duplicate sentiment county months')
    # Use complete feature months only. Missing counties within a complete month
    # are represented explicitly, with train-only feature imputation.
    panel = panel[panel.feature_month.isin(features.feature_month.unique())]
    panel = panel.merge(features, on=['fips', 'feature_month'], how='left', validate='one_to_one')
    panel['sentiment_missing'] = panel.mean_sentiment.isna().astype(float)
    panel['log_post_count'] = np.log1p(panel.post_count.fillna(0))
    panel['log_posts_per_capita'] = np.log1p(100000*panel.post_count.fillna(0)/panel.population)
    previous_features = features[['fips', 'feature_month', 'mean_sentiment']].copy()
    previous_features['prior_feature_month'] = previous_features.feature_month
    previous_features = previous_features.drop(columns='feature_month').rename(columns={'mean_sentiment': 'previous_sentiment'})
    panel['prior_feature_month'] = (pd.PeriodIndex(panel.feature_month, freq='M')-1).astype(str)
    panel = panel.merge(previous_features[['fips', 'prior_feature_month', 'previous_sentiment']],
                        on=['fips', 'prior_feature_month'], how='left', validate='one_to_one')
    panel['sentiment_change'] = panel.mean_sentiment-panel.previous_sentiment
    panel['feature_end'] = pd.PeriodIndex(panel.feature_month, freq='M').end_time.normalize()
    panel['target_start'] = pd.PeriodIndex(panel.month, freq='M').start_time
    panel['target_end'] = pd.PeriodIndex(panel.month, freq='M').end_time.normalize()
    panel['definition_changed'] = (panel.signal != panel.previous_signal).astype(float)
    if (panel.feature_end >= panel.target_start).any():
        raise ValueError('Future feature window')
    return panel.sort_values(['month', 'fips']).reset_index(drop=True)


def fit_predictions(train, test, columns, model, delta=False):
    if train.empty or test.empty:
        raise ValueError('Empty temporal partition')
    if train.target_end.max() >= test.target_start.min():
        raise ValueError('Training target overlaps future test')
    estimator = {'ridge': Ridge(alpha=10),
                 'forest': RandomForestRegressor(n_estimators=250, min_samples_leaf=8,
                                                random_state=42, n_jobs=4),
                 'boosting': HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                                                          l2_regularization=10, random_state=42)}[model]
    pipeline = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), estimator)
    y = train.responsiveness-(train.previous_mask if delta else 0)
    pipeline.fit(train[columns], y)
    result = pipeline.predict(test[columns])+(test.previous_mask.to_numpy() if delta else 0)
    return np.clip(result, 0, 1)


def summarize(predictions):
    metrics = []
    columns = [column for column in predictions if column.endswith('_prediction')]
    for scope, group in [('all', predictions), *list(predictions.groupby('month'))]:
        for column in columns:
            metrics.append({'scope': scope, 'predictor': column.removesuffix('_prediction'),
                'rows': len(group), 'mae': mean_absolute_error(group.responsiveness, group[column]),
                'rmse': np.sqrt(mean_squared_error(group.responsiveness, group[column])),
                'r2': r2_score(group.responsiveness, group[column])})
    return pd.DataFrame(metrics)


def cluster_intervals(predictions, draws=5000):
    """Paired state resampling, conditional on the fitted predictors."""
    rng = np.random.default_rng(91000)
    output = []
    actual = predictions.responsiveness
    for model in ('ridge', 'forest', 'boosting'):
        combined = model+'_combined_delta_prediction'
        for baseline in ('persistence_prediction', model+'_lag_population_delta_prediction',
                         model+'_lag_activity_delta_prediction'):
            improvement = abs(actual-predictions[baseline])-abs(actual-predictions[combined])
            grouped = pd.DataFrame({'state': predictions.geo_group, 'gain': improvement}).groupby('state').gain.agg(['sum', 'count'])
            chosen = rng.integers(len(grouped), size=(draws, len(grouped)))
            samples = grouped['sum'].to_numpy()[chosen].sum(1)/grouped['count'].to_numpy()[chosen].sum(1)
            lo, hi = np.quantile(samples, [.025, .975])
            output.append({'model': model, 'baseline': baseline, 'mae_reduction': improvement.mean(),
                           'ci_low': lo, 'ci_high': hi, 'states': len(grouped)})
    return pd.DataFrame(output)


def evaluate(panel, cutoff, folder):
    train = panel[panel.target_end < pd.Timestamp(cutoff)]
    test = panel[panel.target_start >= pd.Timestamp(cutoff)]
    if train.month.nunique()<2 or test.month.nunique()<1:
        raise ValueError('Need at least two earlier training months and one test month')
    predictions = test[['fips', 'geo_group', 'month', 'signal', 'definition_changed',
                        'responsiveness', 'previous_mask']].copy()
    predictions['persistence_prediction'] = test.previous_mask
    sets = {
        'population': ['log_population'],
        'twitter': ['log_population', 'mean_sentiment', 'positive_share', 'negative_share',
                    'log_post_count', 'log_posts_per_capita', 'sentiment_missing'],
        'lag_population_delta': ['previous_mask', 'log_population'],
        'lag_activity_delta': ['previous_mask', 'log_population', 'log_post_count',
                               'log_posts_per_capita', 'sentiment_missing'],
        'combined_delta': ['previous_mask', 'log_population', 'log_post_count',
                          'log_posts_per_capita', 'mean_sentiment', 'positive_share',
                          'negative_share', 'sentiment_change', 'sentiment_missing']}
    for model in ('ridge', 'forest', 'boosting'):
        for name, columns in sets.items():
            predictions[f'{model}_{name}_prediction'] = fit_predictions(train, test, columns, model, name.endswith('_delta'))
    folder.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(folder/'predictions.csv', index=False)
    summarize(predictions).to_csv(folder/'metrics.csv', index=False)
    cluster_intervals(predictions).to_csv(folder/'state_cluster_intervals.csv', index=False)
    (folder/'manifest.json').write_text(json.dumps({'cutoff': cutoff,
        'training_months': sorted(train.month.unique()), 'test_months': sorted(test.month.unique()),
        'training_rows': len(train), 'test_rows': len(test), 'feature_sets': sets,
        'test_design': 'Fixed earlier training and later held-out target months; counties may repeat across time',
        'limitations': 'Latest revised historical survey data; not verified as-of availability; survey definition changed Feb 2021; exploratory unadjusted state-cluster intervals conditional on fitted models',
        'models': 'Fixed ridge alpha10, forest250/leaf8, boosting150/leaves15/l2=10; no test tuning'}, indent=2))
    print(summarize(predictions).query("scope == 'all'").to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=Path('outputs/multi_month'))
    parser.add_argument('--cutoff', default='2021-01-01')
    parser.add_argument('--exclude-first-week', action='store_true')
    args = parser.parse_args()
    panel = build_panel(args.folder, args.exclude_first_week)
    suffix = '_no_smoothing_overlap' if args.exclude_first_week else ''
    panel.to_csv(args.folder/('features_targets'+suffix+'.csv'), index=False)
    evaluate(panel, args.cutoff, args.folder/('temporal_holdout'+suffix))
    stable = panel[panel.signal == 'smoothed_wwearing_mask']
    evaluate(stable, args.cutoff, args.folder/('stable_definition_holdout'+suffix))


if __name__ == '__main__':
    main()
