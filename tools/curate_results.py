"""Export small, reviewed results; never copy raw tweets or large datasets."""
from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'results'


def copy(source, target):
    source = ROOT / source
    target = DESTINATION / target
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    for name in ['summary.csv', 'manifest.json', 'ideal_information_results.png', 'information_quality_curve.png']:
        copy(f'outputs/ideal_information_benchmark_84d/{name}', f'idealized_grid/{name}')
    for name in ['summary.csv', 'manifest.json', 'trials.csv']:
        copy(f'outputs/matched_moderate_benchmark/confirmation/{name}', f'moderate_diagnostic/{name}')
    copy('outputs/matched_moderate_benchmark/summary.csv', 'moderate_diagnostic/development_grid.csv')
    copy('outputs/matched_moderate_benchmark/reference_calibration.csv', 'moderate_diagnostic/reference_calibration.csv')
    for source, target in [
        ('summary.csv', 'parameter_grid.csv'), ('full_comparison_grid.csv', 'population_control_grid.csv'),
        ('alignment_sensitivity.csv', 'proxy_alignment.csv'), ('cross_survey_agreement.json', 'cross_survey_agreement.json'),
        ('manifest.json', 'manifest.json')]:
        copy(f'outputs/us_proxy_study/geographic/{source}', f'us_geographic/{target}')
    for name in ['summary.csv', 'manifest.json', 'excluded_counties.csv', 'calibration_series.csv']:
        copy(f'outputs/us_proxy_study/real_fall/{name}', f'us_historical/{name}')
    copy('outputs/us_proxy_study/confirmation_summary.csv', 'us_geographic/initial_confirmation.csv')
    copy('outputs/us_proxy_study/moderate/budgets.csv', 'us_moderate/budgets.csv')
    copy('outputs/us_proxy_study/moderate/manifest.json', 'us_moderate/calibration_manifest.json')
    for directory in sorted((ROOT / 'outputs/us_proxy_study/network_aware').iterdir()):
        if not (directory / 'manifest.json').exists():
            continue
        for name in ['summary.csv', 'expected.csv', 'manifest.json', 'optimization.json']:
            copy(directory.relative_to(ROOT) / name, Path('network_aware') / directory.name / name)
    for image in sorted((ROOT / 'outputs/us_proxy_study/figures').glob('*.png')):
        copy(image.relative_to(ROOT), Path('figures') / image.name)
    copy('data/raw/national_hesitancy/source.json', 'provenance/hhs_source.json')
    inventory = []
    for path in sorted(DESTINATION.rglob('*')):
        if path.is_file() and path.name != 'inventory.json':
            content = path.read_bytes()
            inventory.append(dict(path=path.relative_to(DESTINATION).as_posix(), bytes=len(content),
                                  sha256=hashlib.sha256(content).hexdigest()))
    (DESTINATION / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    print(f'Exported {len(inventory)} files, {sum(item["bytes"] for item in inventory):,} bytes.')


if __name__ == '__main__':
    main()
