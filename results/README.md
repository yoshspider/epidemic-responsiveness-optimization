# Reviewed result snapshots

These are derived simulation outputs and figures, not raw datasets.

- `idealized_grid/`: fixed-budget synthetic information, measurement-noise and mobility grid.
- `moderate_diagnostic/`: development grid plus disjoint 64-world confirmation. The budget was calibrated to a moderate uninformed effect; informed outcomes were computed independently. This is a diagnostic, not empirical evidence.
- `us_geographic/`: synthetic initial epidemic states on actual U.S. populations and mobility networks; includes full parameter, population-control and proxy-alignment comparisons.
- `us_historical/`: retrospective 2020 case initialization and transmission calibration. Survey-as-responsiveness assumptions remain unvalidated.
- `us_moderate/`: aggregate-prior budget calibration, without location-specific true response labels.
- `network_aware/`: local numerical horizon planning and paired stochastic evaluations. Each scenario has its own manifest, optimizer diagnostics and comparator-specific effects.
- [`us_weekly_information/`](us_weekly_information/): held-out perfect-information benchmark and fixed-budget survey proxy comparisons on the U.S. network.
- `figures/`: county proxy inputs, spread snapshots, benefit maps and parameter sensitivity.
- `provenance/`: source audit and dataset hashes. No credentials or individual-level records are included.

**Read the baseline column:** an improvement over distribution-only information need not exceed a population-only predictor. Larger gains in selected parameter settings are not universal effects. Confidence intervals condition on the simulated disease/proxy assumptions; they are not causal intervals.

Older separable-planner figures and newer network-aware results are explicitly distinguished in their manifests. No PPO performance claim should be inferred from a static-horizon optimizer result.
