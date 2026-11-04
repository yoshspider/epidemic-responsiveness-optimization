# Epidemic intervention allocation

This workspace contains synthetic mechanism experiments, empirical-input COVID counterfactuals, and a complete September TBCOV sentiment pipeline. See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for evaluated prediction and policy results, source coverage, and interpretation. These simulations do not establish causal intervention effectiveness.

## Run

Use Python 3.11+ and `pip install -r requirements.txt` in your normal virtual environment. In this Codex workspace `run.ps1` uses the bundled Python runtime and local `.deps` dependencies:

```powershell
./run.ps1 --synthetic --counties 24 --days 84 --seeds 5 --output outputs/demo
./run.ps1 --data data/prepared --config config.example.json --days 84 --seeds 20 --output outputs/empirical
```

Portable commands:

```text
python -m unittest discover -s tests -v
python -m epidemic.experiments --synthetic --days 28 --seeds 3
python -m epidemic.prediction features.csv feature_sets.json --folds 5
```

`--synthetic` is mandatory for the debugging generator. Empirical inputs never fall back to synthetic data. Predictions add `--policies none epidemic oracle lookahead predicted` and require a prepared prediction column.

## Empirical input contract

Provide `counties.csv` with one row per county and these columns:

| Column | Meaning |
|---|---|
| fips | Five digit text identifier |
| population_general, population_vulnerable | Counts, with mutually exclusive groups |
| infected_general, infected_vulnerable | Active infections at initialization |
| recovered_general, recovered_vulnerable | Removed counts at initialization |
| responsiveness | Explicitly constructed mask-adherence proxy, in [0,1] |
| cost_per_person | Currency or declared model units per person-equivalent effort |
| predicted_responsiveness | Optional held-out prediction in [0,1] |

Provide `mobility.csv` with `source_fips,destination_fips,weight`. Reject missing population/cost/responsiveness, duplicate counties, and unknown mobility endpoints. Do not silently fill missing counties. A cost quoted per 100,000 people must be divided by 100,000 before ingestion. Record source URLs, download dates, file hashes, geographic vintage, exclusion counts, cost definition, initialization method, and target/feature windows alongside each prepared dataset.

Case reports are not active infections. Initialization requires explicit reporting-rate and infectious-duration assumptions or a fitted observation model; the loader deliberately does not treat cumulative reported cases as I. County equivalents, New York City aggregates, geographic boundary changes, and survey missingness need an explicit crosswalk and coverage report.

## Mathematical conventions

Group order is general then vulnerable. Population is conserved in S+I+R; expected deaths are a weighted outcome, not an additional compartment. Infected people recover with probability `1-exp(-recovery_rate*dt)`. Both transmission coefficients and recovery_rate have units per day. Infection probability is `1-exp(-hazard*dt)`, preventing negative susceptible counts. Weekly steps freeze within-step prevalence, so daily integration is recommended; weekly and daily trajectories need not coincide.

For source group h and target group g, local pressure is `sum_h mixing[g,h]*I[i,h]/N[i]`. Mobility pressure is the same expression after applying W to each source-group pressure. The default all-ones group matrix produces population-weighted homogeneous mixing; unequal coefficients permit within/between-group differences. This matrix is a phenomenological exposure multiplier, not a fitted reciprocal contact survey. Age mixing and IFR values in the example are illustrative parameters requiring calibration.

W has **destination rows and source columns**: `W[i,j]` corresponds to the brief's `M[j,i]`. Remove self edges, sum duplicate edges, and row-normalize incoming weights to one; isolated rows remain zero. W is dimensionless conditional source exposure. `beta_mobility` sets the overall extra intercounty exposure per day. This normalization discards absolute flow intensity: it is appropriate for a relative exposure network, not a claim to recover measured travel contact rates. Mobility truly enters the hazard and the lookahead rollout.

Effort x is person-equivalent coverage effort and costs `x*cost_per_person`. Adoption is `p*(1-exp(-saturation*x/N))`. Effort decays after every step by `retention_daily**dt`; it is a decaying stock, not unique persons reached. The recipient county's adoption reduces both local and imported exposure. Source-side masking is not separately modeled. These are hypotheses to test, not empirical estimates.

Weekly budget releases occur every seven days; unused funds expire. Lump budget is available from the start and persists until spent. Decisions occur at decision_days intervals. If decision intervals exceed seven days, intervening weekly funds can expire without an opportunity to allocate them. Intervention effort persists according to decay in both modes. A final chunk may be smaller than 25,000 when funds are insufficient.

## Policies and fairness

- `none`: no spending.
- `epidemic`: immediate expected marginal objective gain/cost using only an explicitly configured constant prior. The prior defaults to the brief's 0.776 and must be justified from independent training data for empirical work.
- `oracle`: identical immediate greedy calculation with true proxy p.
- `lookahead`: greedy network-wide cumulative objective improvement over horizon_days. Each candidate adds one chunk, then runs a deterministic rollout with no further spending. Horizon truncates at the experiment end.
- `predicted`: identical immediate greedy policy with externally supplied held-out p estimates.

All execution uses the same true p, initial state, cost, objective, disease parameters and budget. Each policy replans from its own evolving state. Planning uses expectations even when execution is stochastic. The lookahead is a limited rollout heuristic: it does not solve the full future budget allocation problem. Greedy with oracle information is not guaranteed to outperform other heuristics over the full epidemic.

Epidemic stochasticity uses inverse-binomial draws with common county/group/time/channel uniforms. Adoption stochasticity draws a county-wide adopter fraction independently each step; this represents transient compliance variability, not persistent individual identities. Infection/recovery draws use pre-step compartments, and new infections cannot recover in the same step. Paired seeds reduce comparison variance without guaranteeing identical realized counts. Deterministic mode repeats have no sampling uncertainty; do not interpret identical-seed confidence intervals as parameter uncertainty.

## Prediction

Feature tables require `fips,responsiveness,geo_group,feature_end,target_start` and numeric covariates. `geo_group` should reflect geographic blocking (state or larger regions); adjacent counties across state borders may require coarser blocks or buffers. All feature windows must finish before survey target_start. One record per county is accepted. Preprocessing is fit inside each training fold. Ridge and random forest generate clipped out-of-fold predictions and held-out MAE/RMSE/R². A feature JSON can contain:

```json
{"demographics": ["age65_share", "income"], "sentiment": ["mean_sentiment", "post_count"], "combined": ["age65_share", "income", "mean_sentiment", "post_count"]}
```

Join the chosen out-of-fold column by FIPS as predicted_responsiveness before running downstream comparisons. Do not replace it with in-sample fitted predictions. Hyperparameter tuning, temporal transfer testing, uncertainty calibration, and spatial buffers are future work. The oracle-recovery fraction is undefined when oracle does not improve over epidemic-only, and can lie outside [0,1].

## Outputs and diagnosis

Each experiment emits metrics, county trajectories, chunk allocations with planned marginal gain/cost, all paired outcome contrasts with t intervals, allocation Jaccard and effort L1 differences, manifest, and an infection plot. Infections and deaths include only incident infections during the window, excluding initial active/removed people. Peak includes initial I. Jaccard describes ever-selected counties and can hide differences in allocation amounts; inspect L1 and chunk logs too.

Reasons for tiny behavior gains include narrow p variation, risk/cost rankings dominating p, correlated p and risk, nearly universal coverage, weak efficacy, limited imported exposure, and short forecast horizons. Check rank changes in marginal gain/cost and allocated amounts before interpreting overlapping selected sets. Do not change parameters solely to force a preferred winner.

For sensitivity experiments rerun the same seeds and initial state while varying budget, efficacy, beta_mobility, horizon_days and objective in configuration files. Controlled behavioral variance/correlation and prediction-error experiments should explicitly label transformed p/predictions as counterfactual. Separate empirical scenarios from synthetic mechanism checks. Include constant p, permuted p, zero mobility and zero efficacy controls. Report paired differences, relative gains, spending and coverage, including negative results.

## Dynamic networks, monthly targets, calibration and Colab

After source clarification, CTIS is the behavior training target; NYT mask-use is an independent geographic comparison. See DATA_SOURCES.md for acquired files, measured coverage and temporal limitations. `python -m epidemic.prepare` audits and transforms downloaded NYT/Census sources. GeoDS and CTIS transformations are available as `geods` and `ctis_monthly` in epidemic.prepare. The CTIS transformation is a descriptive monthly average, not a pooled survey estimator.

Dynamic mobility.csv adds week_start to the existing edge columns; Config.start_date maps dates to simulation days. Every observed week must be present. Forecasts freeze the currently observed network, avoiding access to future network files. Document reporting availability and lag the weekly GeoDS flow series for prospective forecasting.

For county-month prediction panels, use `--test-start YYYY-MM-DD` and include target_end. Training folds exclude all target windows ending on or after that cutoff, and held-out geographic groups are evaluated only on later target windows. Rows outside the evaluation period have missing predictions; select a specific evaluation window before joining predictions into the simulator. Geographic blocking alone does not remove adjacent-border correlations; buffers may still be needed.

`epidemic.calibration.calibrate` fits beta_local and beta_mobility on training incident reports with a held-out time window, fixed externally specified reporting fraction/delay and fixed recovery/contact assumptions. It flags ill-conditioning through the Jacobian condition number. It has been validated on simulated known rates, not fitted to the downloaded empirical cases. Treat negative report revisions/gaps explicitly before fitting. Reporting rates and disease transmission may be weakly identifiable, and the simple observation model does not estimate changing test availability or reporting delays.

`python -m epidemic.sensitivity --synthetic --days 28 --seeds 5` runs controlled one-at-a-time budget, efficacy, mobility, horizon, objective, behavioral variance/correlation and prediction-noise checks. It labels transformations as counterfactual.

`colab/epidemic_research.ipynb` and `colab/research_source.zip` are prepared for manual upload to Google Colab. Optional Config.rollout_backend supports numpy (default), auto (CUDA if available, otherwise NumPy), cuda (requires CUDA), or torch_cpu. Install PyTorch separately for tensor backends; Colab commonly supplies it. Tensor rollouts use float64 sparse matrices and are checked against NumPy on CPU; CUDA parity checks run when hardware is available. CUDA has not been tested locally. Benchmark GPU speed on your actual network before claiming acceleration. Uploading or running in your Google account has not been performed.

## Performance and remaining research work

Epidemic updates and immediate candidate scores are vectorized, and mobility uses sparse CSR matrices. Lookahead candidates are processed in configurable batches. Every county is evaluated: no undocumented candidate filtering. Its cost is roughly O(chunks × counties × horizon × network-update-cost), and exhaustive national lookahead may be slow. The optional torch backend batches rollouts on CPU/CUDA; actual speed depends on graph density and batch size. RL is not implemented. A local synthetic 3,126-county ring-network benchmark measured 84 epidemic steps at ~0.03 seconds and one exhaustive 28-day lookahead chunk at ~64 seconds on NumPy. This is not a benchmark of the much denser empirical GeoDS graph. The runner holds one policy trajectory at a time, but county/time output remains substantial.

Before paper-ready conclusions: obtain and review empirical data; prepare crosswalks and provenance; fit and validate disease/contact parameters and observation model on separate windows; justify intervention costs, proxy validity and persistence; run geographic/temporal prediction validation; conduct preregistered sensitivity and paired stochastic analyses; profile national lookahead and add validated acceleration if needed. This scaffold cannot establish causal intervention responsiveness from a cross-sectional adherence survey.
