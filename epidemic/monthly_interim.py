"""Exploratory December holdout from the first three complete feature months.

Separate from the fixed January-onward primary analysis; no model tuning.
"""
from pathlib import Path
import json
from .monthly_prediction import build_panel, evaluate


def main():
    source = Path('outputs/multi_month')
    for month in ['2020-09', '2020-10', '2020-11']:
        if not (source/f'sentiment_{month}.complete.json').exists():
            raise ValueError('Interim requires complete September–November aggregates')
    for exclude, suffix in [(False, ''), (True, '_no_smoothing_overlap')]:
        panel = build_panel(source, exclude)
        panel = panel[panel.month.between('2020-10', '2020-12')]
        output = source/('interim_december'+suffix)
        output.mkdir(exist_ok=True)
        panel.to_csv(output/'features_targets.csv', index=False)
        evaluate(panel, '2020-12-01', output)
        manifest = json.loads((output/'manifest.json').read_text())
        manifest['analysis_status'] = 'EXPLORATORY INTERIM; October–November training, December test; fixed hyperparameters; separate from primary January-onward analysis'
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
