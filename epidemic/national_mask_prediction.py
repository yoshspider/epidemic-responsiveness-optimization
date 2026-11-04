"""Portable, state-held-out mask predictors used by the national experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    out = Path('outputs/nyt_alternative')
    out.mkdir(parents=True, exist_ok=True)
    mask = pd.read_csv('data/raw/nyt_mask_use.csv', dtype={'COUNTYFP': str})
    mask['fips'] = mask.COUNTYFP.str.zfill(5)
    mask['truth'] = mask.ALWAYS + mask.FREQUENTLY + .5 * mask.SOMETIMES
    population = pd.read_csv('data/transformed/census_population_2019.csv', dtype={'fips': str})
    svi = pd.read_csv('data/raw/alternative_sources/cdc_svi2018_counties.csv', dtype={'FIPS': str})
    svi['fips'] = svi.FIPS.str.zfill(5)
    columns = ['E_PCI', 'EP_POV', 'EP_UNEMP', 'EP_NOHSDP', 'EP_AGE65', 'EP_DISABL',
               'EP_SNGPNT', 'EP_LIMENG', 'EP_NOVEH', 'EP_UNINSUR', 'AREA_SQMI']
    for column in columns:
        svi[column] = pd.to_numeric(svi[column], errors='coerce').where(lambda values: values >= 0)
    frame = mask[['fips', 'truth']].merge(
        population[['fips', 'population']], on='fips', validate='one_to_one')
    frame = frame.merge(svi[['fips'] + columns], on='fips', how='left', validate='one_to_one')
    frame['log_population'] = np.log(frame.population)
    frame['log_density'] = np.log1p(frame.population / frame.AREA_SQMI.replace(0, np.nan))
    feature_sets = {'population': ['log_population'],
                    'demographics': ['log_population', 'log_density'] + columns[:-1]}
    quality = []
    for model in ['ridge', 'forest']:
        for name, features in feature_sets.items():
            predictions = np.zeros(len(frame))
            for train, test in GroupKFold(5).split(frame, groups=frame.fips.str[:2]):
                estimator = Ridge(alpha=10) if model == 'ridge' else RandomForestRegressor(
                    n_estimators=250, min_samples_leaf=8, random_state=42, n_jobs=2)
                pipeline = make_pipeline(
                    SimpleImputer(
                        keep_empty_features=True),
                    StandardScaler(),
                    estimator)
                pipeline.fit(frame.iloc[train][features], frame.iloc[train].truth)
                predictions[test] = np.clip(pipeline.predict(frame.iloc[test][features]), 0, 1)
            frame[f'{model}_{name}'] = predictions
            quality.append(dict(model=model, features=name, counties=len(frame),
                                mae=mean_absolute_error(frame.truth, predictions),
                                r2=r2_score(frame.truth, predictions)))
    frame.to_csv(out / 'state_heldout_predictions.csv', index=False)
    pd.DataFrame(quality).to_csv(out / 'prediction_metrics.csv', index=False)
    print(pd.DataFrame(quality).to_string(index=False))


if __name__ == '__main__':
    main()
