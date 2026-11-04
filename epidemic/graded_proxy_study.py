"""Retrospective COVID replay with the brief's graded NYT behavior proxy."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .observed_study import prepare
from .model import Config, Model
from .experiments import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('outputs/graded_proxy_replay'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counties, n, _, initial, mobility, _, audit = prepare(Path('data/transformed'))
    raw = pd.read_csv('data/raw/nyt_mask_use.csv', dtype={'COUNTYFP': str}).set_index('COUNTYFP')
    mask = raw.reindex(counties.fips)
    truth = (mask.ALWAYS + mask.FREQUENTLY + .5*mask.SOMETIMES).to_numpy()
    if not np.isfinite(truth).all():
        raise ValueError('Missing graded proxy')
    rates = pd.read_csv('outputs/temporal_replay_fixed_share/weekly_rates.csv')
    schedule = tuple(map(tuple, rates[['day', 'beta_local', 'beta_mobility']].to_numpy()))
    manifest = {'data_kind': 'RETROSPECTIVE EMPIRICAL-INPUT COUNTERFACTUAL',
        'proxy': 'ALWAYS + FREQUENTLY + 0.5*SOMETIMES; descriptive adherence, not identified causal response',
        'proxy_mean': float(truth.mean()), 'proxy_sd': float(truth.std()),
        'input_audit': audit, 'rate_source': 'outputs/temporal_replay_fixed_share/weekly_rates.csv',
        'rate_limitation': 'Contemporaneously fitted retrospective rates; fixed 15% mobility share; not prospective holdout',
        'budgets': [.005, .02], 'seeds': list(range(85000, 85000+args.seeds)),
        'assumptions': 'cost=1 model unit/person effort, efficacy=.5, retention=.95, fixed July mobility; aggregate proxy mean supplied equally',
        'uncertainty': 'Paired epidemic noise only; not parameter, causal, or survey uncertainty'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for fraction in manifest['budgets']:
        config = Config(prior=float(truth.mean()), mean_prior=float(truth.mean()),
                        budget=fraction*float(n.sum()), stochastic=True,
                        beta_schedule=schedule, start_date='2020-08-31')
        model = Model(n, truth, np.ones(len(n)), mobility, config)
        manifest[f'config_{fraction}'] = asdict(config)
        for seed in manifest['seeds']:
            for policy in ('epidemic', 'oracle'):
                metrics, *_ = run(model, initial, policy, 84, seed, record=False)
                rows.append({'budget_fraction': fraction, **metrics})
            pd.DataFrame(rows).to_csv(args.output/'trials.csv', index=False)
            print(f'graded COVID proxy budget {fraction}, seed {seed}', flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    summaries = []
    for fraction, group in pd.DataFrame(rows).groupby('budget_fraction'):
        wide = group.pivot(index='seed', columns='policy', values='infections')
        delta = wide.epidemic-wide.oracle
        half = t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
        summaries.append({'budget_fraction': fraction, 'additional_infections_prevented': delta.mean(),
                          'ci_low': delta.mean()-half, 'ci_high': delta.mean()+half,
                          'relative_percent': 100*delta.mean()/wide.epidemic.mean()})
    pd.DataFrame(summaries).to_csv(args.output/'comparisons.csv', index=False)


if __name__ == '__main__':
    main()
