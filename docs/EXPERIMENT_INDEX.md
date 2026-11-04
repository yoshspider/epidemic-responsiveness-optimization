# Experiment index

## Current entry points

| Purpose | Module |
|---|---|
| Fixed-budget synthetic information/noise/mobility grid | `ideal_information_benchmark` |
| Moderate-effect development grid | `matched_moderate_benchmark` |
| Fresh 64-world moderate-effect confirmation | `moderate_ideal_confirm` |
| Public HHS county-survey acquisition | `national_hesitancy_source` |
| September–November weekly mobility acquisition | `us_proxy_data` |
| Historical initialization and retrospective calibration | `us_proxy_prepare` |
| Population-weighted static horizon allocation/SIR | `us_proxy_core` |
| Network-aware constrained allocation and exact adjoint | `us_proxy_network` |
| Geographic and historical proxy experiments | `us_proxy_study`, `us_proxy_network_study` |
| Population/state controls and alignment sensitivity | `us_proxy_controls`, `us_proxy_control_grid` |
| Paired epidemic-noise evaluation | `us_proxy_confirmation` |
| County maps and sensitivity figures | `us_proxy_figures` |
| Budget calibration from aggregate information only | `us_moderate_calibrate` |

Use `python -m epidemic.<module>`. Most empirical modules expect separately downloaded
public inputs under `data/raw/` and the intermediate files named in their manifests.
No raw dataset is needed for the synthetic benchmark or unit tests.

## PPO and prediction

`proposal_ppo` implements the weekly exponential intervention model and PPO.
`national_ppo_train` and `national_ppo_evaluate` retain the historical PPO evaluation;
`anchored_ppo` is an exploratory residual-policy variant. Install the `rl` extra for these.
The current static-horizon results do **not** claim PPO achieved the same improvements.

`prediction`, `monthly_prediction`, `midterm_prediction`, and `alternative_prediction`
implement geographic/temporal blocking. `monthly_download`, `monthly_sentiment`, and
`tbcov_pipeline` retain the raw-archive and county-aggregate TBCOV workflows.

## Earlier experiments

Earlier modules are retained for provenance, not as the default model. In particular,
`model`, `experiments`, `planning`, and `policies` use the older capped-adoption/effort-stock
framework. Its effect sizes must not be mixed with the exponential-response experiments.

The documents under `docs/archive/` are historical snapshots with older assumptions
and incomplete-stage descriptions. They are superseded by the top-level README and
the manifests attached to the exported results.
