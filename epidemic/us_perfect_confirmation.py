"""Aggregate disjoint response-world confirmation without retuning policies."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    root = Path('outputs/us_perfect_information')
    out = root / 'confirmation'
    out.mkdir(exist_ok=True)
    frames = []
    hashes = set()
    settings = []
    for seed in [1011000, 1011004, 1011008, 1011012]:
        folder = root / f'recipient_0.36_0.8_0.15_{seed}_stratified_weekly'
        meta = json.loads((folder / 'manifest.json').read_text())
        if meta['phase'] != 'confirmation' or meta['worlds'] != 4:
            raise ValueError('Incomplete or mislabeled held-out confirmation')
        frame = pd.read_csv(folder / 'trials.csv')
        frame['world'] = frame['draw'] + seed
        frames.append(frame)
        baseline = np.load(folder / 'selected_baseline.npz')['dose']
        hashes.add(hashlib.sha256(baseline.tobytes()).hexdigest())
        settings.append(meta)
    if len(hashes) != 1:
        raise ValueError('Uninformed comparator changed between held-out batches')
    table = pd.concat(frames, ignore_index=True)
    if table.duplicated(['world', 'policy']).any() or table.world.nunique() != 16:
        raise ValueError('World duplication or missing confirmation cases')
    wide = table.pivot(index='world', columns='policy', values='infections')
    if (wide.perfect > wide.uninformed + 1e-4).any():
        raise ValueError('Perfect-information reference failed feasible-baseline safeguard')
    rows = []
    rng = np.random.default_rng(1019000)
    bootstrap = rng.integers(0, len(wide), size=(10000, len(wide)))
    for policy in wide:
        values = wide[policy]
        gain = wide.uninformed - values
        half = t.ppf(.975, 15) * gain.std(ddof=1) / 4
        reduction = 100 * (wide.none - values) / wide.none
        interval = 100 * gain.to_numpy()[bootstrap].mean(1) / \
            wide.uninformed.to_numpy()[bootstrap].mean(1)
        rows.append(dict(policy=policy, infections=values.mean(), prevention_percent=reduction.mean(),
                         extra_avoided=gain.mean(), gain_ci_low=gain.mean() - half, gain_ci_high=gain.mean() + half,
                         relative_reduction_percent=100 * gain.mean() / wide.uninformed.mean(),
                         relative_ci_low=float(np.quantile(interval, .025)), relative_ci_high=float(np.quantile(interval, .975)),
                         worst_world_relative_percent=float((100 * gain / wide.uninformed).min()),
                         best_world_relative_percent=float((100 * gain / wide.uninformed).max())))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'summary.csv', index=False)
    table.to_csv(out / 'trials.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(dict(phase='HELD-OUT CONFIRMATION; no empirical proxy tested',
                                                       response_worlds=16, seeds='1011000–1011015, disjoint from development and prior calibration',
                                                       mechanism='Recipient exponential attenuation exp(-p*d); original proposal form',
                                                       p='Independent Uniform(0,1) by county, fixed over time; no survey values or rank transformations',
                                                       data='Actual Census populations and twelve observed weekly GeoDS networks; synthetic initial epidemic; not a historical COVID forecast',
                                                       beta=.36, mobility=.15, mean_dose_per_week=.8, days=84, schedule='Twelve weekly resource allocations planned over full horizon; same budget each week',
                                                       baseline='Frozen ensemble-optimized uninformed weekly plan, selected using eight independent prior validation worlds; identical in every confirmation batch',
                                                       baseline_array_sha256=next(iter(hashes)), prior_scheme='County-stratified Uniform simulation, eight training and eight validation maps',
                                                       optimizer='Static80-iteration initialization plus80-iteration weekly refinement; finite local optimization, not global optimality or PPO',
                                                       uncertainty='Paired t intervals for absolute gains and10000-paired-world bootstrap intervals for relative gains; uncertainty over synthetic response assignments only',
                                                       limitation='No causal responsiveness estimate or empirical proxy effectiveness claim'), indent=2))
    print(result.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
