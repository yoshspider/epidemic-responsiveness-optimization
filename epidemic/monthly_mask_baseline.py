"""Evaluate actual survey persistence while Twitter archives are downloading."""
from pathlib import Path
import numpy as np
import pandas as pd
from .monthly_prediction import fit_predictions, summarize


def main():
    folder = Path('outputs/multi_month')
    masks = pd.read_csv(folder/'mask_county_month.csv', dtype={'fips': str})
    masks = masks[~((masks.month >= '2021-02') & (masks.signal == 'smoothed_wwearing_mask'))]
    masks = masks[masks.observed_days >= 20].copy()
    masks['feature_month'] = (pd.PeriodIndex(masks.month, freq='M')-1).astype(str)
    previous = masks[['fips', 'month', 'responsiveness', 'signal']].rename(columns={
        'month': 'feature_month', 'responsiveness': 'previous_mask', 'signal': 'previous_signal'})
    panel = masks.merge(previous, on=['fips', 'feature_month'], validate='one_to_one')
    population = pd.read_csv('data/transformed/census_population_2019.csv', dtype={'fips': str})
    panel = panel.merge(population[['fips', 'population']], on='fips', validate='many_to_one')
    panel['log_population'] = np.log(panel.population)
    panel['geo_group'] = panel.fips.str[:2]
    panel['target_start'] = pd.PeriodIndex(panel.month, freq='M').start_time
    panel['target_end'] = pd.PeriodIndex(panel.month, freq='M').end_time.normalize()
    panel['definition_changed'] = (panel.signal != panel.previous_signal).astype(float)
    train = panel[panel.month < '2021-01']
    test = panel[panel.month >= '2021-01'].copy()
    columns = ['fips', 'geo_group', 'month', 'signal', 'definition_changed', 'responsiveness', 'previous_mask']
    predictions = test[columns].copy()
    predictions['persistence_prediction'] = test.previous_mask
    for model in ('ridge', 'forest', 'boosting'):
        predictions[model+'_population_prediction'] = fit_predictions(train, test, ['log_population'], model)
        predictions[model+'_lag_population_prediction'] = fit_predictions(train, test, ['previous_mask', 'log_population'], model, delta=True)
    predictions.to_csv(folder/'mask_baseline_predictions.csv', index=False)
    metrics = summarize(predictions)
    metrics.to_csv(folder/'mask_baseline_metrics.csv', index=False)
    intervals = []
    rng = np.random.default_rng(92000)
    for scope, group in [('all', predictions), *list(predictions.groupby('month'))]:
        for model in ('ridge', 'forest', 'boosting'):
            gain = abs(group.responsiveness-group.persistence_prediction)-abs(group.responsiveness-group[model+'_lag_population_prediction'])
            grouped = pd.DataFrame({'state': group.geo_group, 'gain': gain}).groupby('state').gain.agg(['sum', 'count'])
            chosen = rng.integers(len(grouped), size=(5000, len(grouped)))
            samples = grouped['sum'].to_numpy()[chosen].sum(1)/grouped['count'].to_numpy()[chosen].sum(1)
            lo, hi = np.quantile(samples, [.025, .975])
            intervals.append({'scope': scope, 'model': model, 'mae_reduction_vs_persistence': gain.mean(),
                              'ci_low': lo, 'ci_high': hi, 'states': len(grouped)})
    pd.DataFrame(intervals).to_csv(folder/'mask_baseline_state_intervals.csv', index=False)
    panel.groupby(['month', 'signal']).agg(counties=('fips', 'size'),
        masking_mean=('responsiveness', 'mean'), masking_sd=('responsiveness', 'std')).to_csv(folder/'mask_panel_coverage.csv')
    print(metrics.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
