# Reproduce the weekly U.S. experiment

Check out the [midterm archive at commit `02695ee`](https://github.com/yoshspider/epidemic-responsiveness-optimization/tree/02695ee) before using the commands below. The current branch provides the [case-calibrated replay](../case_calibrated_proxy/README.md#reproduce).

This guide starts from a fresh checkout. Run all commands at the repository root. The published CSVs and plots are in this folder; reruns write to ignored `data/` and `outputs/`. Keep the published tables as references when checking a rerun. No GPU, PPO training, TBCOV download, or historical case fitting is needed for this experiment.

## 1. Set up Python

Use Python 3.12 if available (the project supports 3.11 or newer). Commands below use Bash, such as Git Bash on Windows. If using PowerShell, activate with `.venv\Scripts\Activate.ps1` and replace the environment exports with `$env:OPENBLAS_NUM_THREADS='1'` and `$env:OMP_NUM_THREADS='1'`.

```bash
python -m venv .venv
# Bash on Linux/macOS:
source .venv/bin/activate
# Git Bash on Windows: source .venv/Scripts/activate
python -m pip install -e .
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m unittest discover -s tests -p test_responsiveness_mechanisms.py -v
```

For an environment record, save `python --version` and `python -m pip freeze`. The reference workspace used Python 3.12, NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2, Pillow 12.3.0 and pyshp 3.1.6. Optimizer stopping points and plots can differ slightly across numerical library versions and platforms.

## 2. Download and prepare the inputs

```bash
python -m epidemic.bootstrap_us --mobility
python -m epidemic.us_proxy_study --prepare-only
python -m epidemic.us_weekly_ctis_inputs
```

- The bootstrap downloads Census populations and boundaries, July NYT masking, HHS estimates and twelve GeoDS weeks. It also downloads NYT cases and CDC SVI for other workflows; those files do not enter this weekly model.
- `--prepare-only` stops after creating `outputs/us_proxy_study/geographic/`. This folder must contain `counties.csv`, `initial.npz`, `proxy_values.csv` and `mobility_0.npz` through `mobility_11.npz`. Check for **3,142 counties, 328,239,523 residents**, and matching FIPS order across arrays. If you change county geography, use a fresh output directory tree so cached matrices cannot be reused with a new ordering.
- The CTIS helper gets October 2020 daily masking and February 2021 daily vaccine willingness. It writes the October county-month aggregate to `data/transformed/ctis_october2020_monthly_proxy.csv` and leaves the February daily export at `data/raw/vaccine_experiment/acceptance_2021-02.csv`. Follow-up models keep counties with missing CTIS measurements; they do not filter the network to survey respondents.

Archived survey APIs can change or return revised rows. The helper records source URLs and hashes in `data/raw/weekly_ctis_sources.json`. If an endpoint is unavailable, place the matching archived CSV at the path used by the helper and rerun it. The two CTIS files used for the published run have these SHA-256 fingerprints:

| Raw file | SHA-256 |
|---|---|
| `ctis_mask_october2020_daily.csv` | `6a1c512acd59fa0d0954e5daaf7dd28648420a2cebf19cea0633a587529f8686` |
| `vaccine_experiment/acceptance_2021-02.csv` | `3e75da021215bc714a78c387e00ac424fe7b130af54da7071454d8d23a3eb382` |

## 3. Create and freeze the uninformed schedule

Run both development stages. The first uses a static schedule as an initializer; the second produces the weekly schedule. These use separate prior training and validation maps. Keep the frozen schedule for all confirmation batches.

```bash
ROOT=outputs/us_perfect_information
STATIC=$ROOT/recipient_0.36_0.8_0.15_1010000_stratified
WEEKLY=${STATIC}_weekly
python -m epidemic.us_perfect_information --mechanism recipient --beta .36 --dose .8 --mobility .15 --prior-scheme stratified --iterations 80
python -m epidemic.us_perfect_information --mechanism recipient --beta .36 --dose .8 --mobility .15 --prior-scheme stratified --iterations 80 --schedule weekly --warmstart-from "$STATIC"
```

Inspect `$WEEKLY/manifest.json` and `$WEEKLY/selected_baseline.npz`. This optimizer caches some plans in the scenario directory; use fresh scenario outputs if changing parameters, data, or code. Do not retune on the confirmation seeds.

## 4. Confirm perfect information on 16 independent maps

```bash
for seed in 1011000 1011004 1011008 1011012; do
  python -m epidemic.us_perfect_information --mechanism recipient --beta .36 --dose .8 --mobility .15 --prior-scheme stratified --iterations 80 --schedule weekly --phase confirmation --worlds 4 --seed-start "$seed" --prior-from "$WEEKLY"
done
python -m epidemic.us_perfect_confirmation
python -m epidemic.us_perfect_stress
```

The aggregator checks that the four batches used identical baseline arrays and 16 unique worlds. Compare `outputs/us_perfect_information/confirmation/summary.csv` with [perfect_summary.csv](perfect_summary.csv). Expect about 299.53M infections without intervention, 188.34M uninformed, and 155.25M with perfect information. Stress tests reuse the planned schedules and change one assumption at a time.

## 5. Run the survey comparison and render

```bash
python -m epidemic.us_weekly_proxy
python -m epidemic.us_weekly_proxy_followup
python -m epidemic.us_weekly_proxy_followup --acceptance data/raw/vaccine_experiment/acceptance_2021-02.csv
python -m epidemic.us_weekly_proxy_variation
python -m epidemic.us_weekly_proxy_figures
python -m epidemic.us_weekly_spread_atlas
python -m epidemic.us_weekly_spread_animation
```

| Checkpoint under `outputs/us_weekly_proxy/` | What to expect |
|---|---|
| `summary.csv` | Both percentile and raw scenarios; compare with [proxy_summary.csv](proxy_summary.csv) |
| `acceptance_followup.csv` | Best tested percentile plan: `acceptance_population_mean`, about **90.38M** modeled infections |
| `candidate_variation.csv` | 515 October masking counties, 475 February willingness counties, 462 counties on common survey support |
| `spread/week_01.png` through `week_12.png` | Seven policies per week; the renderer checks that county totals match the outcome tables |
| `weekly_spread.gif` | Twelve frames in order, looping with a pause at the end |

The weekly map color is **incident** infections per 100,000; the summary is the **cumulative** number of new infections over the horizon. The raw-score sensitivity changes both the mean and spread of the target. See [METHODS.md](METHODS.md) before interpreting the proxy estimates or changing the model.
